"""
WebSocket-based online matchmaking with server-side move validation.

Protocol:
  Client → Server:  { type: 'move', from_r, from_c, to_r, to_c }
  Server → Client:  { type: 'state', state: { boardState, currentTurn,
                        actionsRemaining, bonusPending, moveCount,
                        gameOver, gameWinner } }
  Invalid move:     server sends back current state (reverts optimistic update)
  Game over:        server detects it, calls _update_elo, broadcasts final state
"""

import sys, os, asyncio, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'engine'))

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from jose import jwt

from ..database import SessionLocal
from ..models import User, Match, GameChallenge
from ..auth import SECRET_KEY, ALGORITHM
from game import GameState, get_valid_moves   # noqa: E402

router = APIRouter()

matchmaking_queue: list = []
active_games: dict      = {}
challenge_games: dict   = {}   # game_token → {user, ws, entry}
_counter                = 0


# ── Helpers ────────────────────────────────────────────────────

def _board_to_json(board):
    return [
        [
            {"player": p.player, "isVeylant": p.is_veylant,
             "hasCrossed": p.has_crossed, "hasEverCrossed": p.has_ever_crossed}
            if p else None
            for p in row
        ]
        for row in board
    ]


def _gs_to_state(gs: GameState) -> dict:
    return {
        "boardState":       _board_to_json(gs.board),
        "currentTurn":      gs.current_player,
        "actionsRemaining": gs.actions_left,
        "bonusPending":     gs.bonus_pending,
        "moveCount":        gs.move_count,
        "gameOver":         gs.game_over,
        "gameWinner":       gs.winner,
    }


async def _get_user(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        uid = int(payload["sub"])
        db  = SessionLocal()
        u   = db.query(User).filter(User.id == uid).first()
        db.close()
        return {"id": u.id, "username": u.username, "elo": round(u.elo)} if u else None
    except Exception:
        return None


async def _update_elo(gid: str, winner: int):
    game = active_games.pop(gid, None)
    if not game:
        return
    is_rated = game.get("rated", True)
    db = SessionLocal()
    try:
        p1 = db.query(User).filter(User.id == game["p1"]["id"]).first()
        p2 = db.query(User).filter(User.id == game["p2"]["id"]).first()
        if not (p1 and p2):
            return

        if winner == 1:
            s1, s2 = 1.0, 0.0; p1.wins   += 1; p2.losses += 1; r1, r2 = 'win',  'loss'
        elif winner == 2:
            s1, s2 = 0.0, 1.0; p1.losses += 1; p2.wins   += 1; r1, r2 = 'loss', 'win'
        else:
            s1, s2 = 0.5, 0.5; p1.draws  += 1; p2.draws  += 1; r1, r2 = 'draw', 'draw'

        p1_elo_before = round(p1.elo)
        p2_elo_before = round(p2.elo)
        p1_elo_after  = p1_elo_before
        p2_elo_after  = p2_elo_before

        if is_rated:
            K    = 32
            exp1 = 1 / (1 + 10 ** ((p2.elo - p1.elo) / 400))
            p1.elo = max(100, round(p1.elo + K * (s1 - exp1)))
            p2.elo = max(100, round(p2.elo + K * (s2 - (1 - exp1))))
            p1_elo_after = round(p1.elo)
            p2_elo_after = round(p2.elo)

        db.add(Match(
            player1_id    = p1.id,
            player2_id    = p2.id,
            result_p1     = r1,
            elo_p1_before = p1_elo_before,
            elo_p1_after  = p1_elo_after,
            elo_p2_before = p2_elo_before,
            elo_p2_after  = p2_elo_after,
            is_bot        = False,
            rated         = is_rated,
        ))
        db.commit()
    finally:
        db.close()


# ── Per-player relay with server-side validation ───────────────

async def _relay(gid: str, my_ws: WebSocket, player_num: int):
    """
    Handles incoming messages from one player.
    Validates moves against the server's authoritative GameState,
    applies them, and broadcasts the new state to both players.
    """
    try:
        while True:
            raw  = await my_ws.receive_text()
            data = json.loads(raw)
            t    = data.get("type")

            if t == "resign":
                game = active_games.get(gid)
                if game:
                    winner = 2 if player_num == 1 else 1
                    opp_ws = game["p1"]["ws"] if player_num == 2 else game["p2"]["ws"]
                    try:
                        await opp_ws.send_json({"type": "opponent_resigned"})
                    except Exception:
                        pass
                    await _update_elo(gid, winner)
                return

            if t != "move":
                continue   # ignore unknown messages

            game = active_games.get(gid)
            if not game:
                break

            gs: GameState = game["gs"]

            # ── 1. Validate turn ──
            if gs.current_player != player_num:
                await my_ws.send_json({"type": "state", "state": _gs_to_state(gs)})
                continue

            # ── 2. Parse coordinates ──
            try:
                from_r = int(data["from_r"])
                from_c = int(data["from_c"])
                to_r   = int(data["to_r"])
                to_c   = int(data["to_c"])
            except (KeyError, ValueError, TypeError):
                await my_ws.send_json({"type": "state", "state": _gs_to_state(gs)})
                continue

            # ── 3. Validate piece ownership ──
            piece = gs.board[from_r][from_c] if (0 <= from_r < 20 and 0 <= from_c < 11) else None
            if piece is None or piece.player != player_num:
                await my_ws.send_json({"type": "state", "state": _gs_to_state(gs)})
                continue

            # ── 4. Find matching legal move ──
            valid_moves = get_valid_moves(gs.board, from_r, from_c)
            move_match  = next(
                (m for m in valid_moves if tuple(m["target"]) == (to_r, to_c)),
                None
            )
            if move_match is None:
                await my_ws.send_json({"type": "state", "state": _gs_to_state(gs)})
                continue

            # ── 5. Apply move (immutable — returns new GameState) ──
            async with game["lock"]:
                new_gs     = gs.apply(from_r, from_c, move_match)
                game["gs"] = new_gs

            state_msg = {"type": "state", "state": _gs_to_state(new_gs)}

            # ── 6. Broadcast to both players ──
            opp_ws = game["p1"]["ws"] if player_num == 2 else game["p2"]["ws"]
            try:
                await opp_ws.send_json(state_msg)
            except Exception:
                pass
            await my_ws.send_json(state_msg)

            # ── 7. Game over? ──
            if new_gs.game_over:
                await _update_elo(gid, new_gs.winner or 0)
                return

    except WebSocketDisconnect:
        game = active_games.get(gid)
        if game:
            opp_ws = game["p1"]["ws"] if player_num == 2 else game["p2"]["ws"]
            try:
                await opp_ws.send_json({"type": "opponent_left"})
            except Exception:
                pass
    finally:
        active_games.pop(gid, None)


# ── Matchmaking endpoint ───────────────────────────────────────

@router.websocket("/ws/play")
async def ws_play(websocket: WebSocket, token: str = Query(...),
                  challenge_token: str = Query(None),
                  rated: str = Query("true")):
    await websocket.accept()

    user = await _get_user(token)
    if not user:
        await websocket.send_json({"type": "error", "msg": "Unauthorized"})
        await websocket.close()
        return

    global _counter, matchmaking_queue

    # ── Challenge (direct invite) path ──────────────────────────────
    if challenge_token:
        db = SessionLocal()
        try:
            ch = db.query(GameChallenge).filter(
                GameChallenge.game_token == challenge_token,
                GameChallenge.status.in_(["pending", "accepted"])
            ).first()
            if not ch or user["id"] not in (ch.challenger_id, ch.challenged_id):
                await websocket.send_json({"type": "error", "msg": "Challenge not found"})
                await websocket.close()
                return
            is_rated = ch.rated
        finally:
            db.close()

        if challenge_token in challenge_games:
            # Partner is already waiting — pair immediately
            partner = challenge_games.pop(challenge_token)
            _counter += 1
            gid = f"g{_counter}"

            active_games[gid] = {
                "p1":   {**partner["user"], "ws": partner["ws"]},
                "p2":   {**user, "ws": websocket},
                "gs":   GameState(),
                "lock": asyncio.Lock(),
                "rated": is_rated,
            }
            partner["entry"]["game_id"] = gid
            initial_state = _gs_to_state(active_games[gid]["gs"])

            await partner["ws"].send_json({
                "type": "matched", "game_id": gid, "player": 1,
                "opponent": user["username"], "opponent_elo": user["elo"],
                "rated": is_rated, "state": initial_state,
            })
            await websocket.send_json({
                "type": "matched", "game_id": gid, "player": 2,
                "opponent": partner["user"]["username"], "opponent_elo": partner["user"]["elo"],
                "rated": is_rated, "state": initial_state,
            })
            await _relay(gid, websocket, 2)

        else:
            # Wait for partner
            entry = {"game_id": None}
            challenge_games[challenge_token] = {"user": user, "ws": websocket, "entry": entry}
            await websocket.send_json({"type": "waiting_for_opponent"})

            try:
                while entry["game_id"] is None:
                    try:
                        raw  = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                        data = json.loads(raw)
                        if data.get("type") == "cancel":
                            challenge_games.pop(challenge_token, None)
                            await websocket.send_json({"type": "cancelled"})
                            return
                    except asyncio.TimeoutError:
                        pass
            except WebSocketDisconnect:
                challenge_games.pop(challenge_token, None)
                return

            gid  = entry["game_id"]
            game = active_games.get(gid)
            if not game:
                return
            await _relay(gid, websocket, 1)

        return  # challenge path handled

    # ── Regular matchmaking path ────────────────────────────────────
    is_rated = rated.lower() != "false"
    matchmaking_queue = [q for q in matchmaking_queue if q["id"] != user["id"]]

    if matchmaking_queue:
        # ── Matched immediately ──────────────────────────────────────
        opp = matchmaking_queue.pop(0)
        _counter += 1
        gid = f"g{_counter}"

        active_games[gid] = {
            "p1":   {**opp,  "ws": opp["ws"]},
            "p2":   {**user, "ws": websocket},
            "gs":   GameState(),
            "lock": asyncio.Lock(),
            "rated": is_rated,
        }
        opp["game_id"] = gid

        initial_state = _gs_to_state(active_games[gid]["gs"])

        await opp["ws"].send_json({
            "type": "matched", "game_id": gid, "player": 1,
            "opponent": user["username"], "opponent_elo": user["elo"],
            "rated": is_rated, "state": initial_state,
        })
        await websocket.send_json({
            "type": "matched", "game_id": gid, "player": 2,
            "opponent": opp["username"], "opponent_elo": opp["elo"],
            "rated": is_rated, "state": initial_state,
        })

        await _relay(gid, websocket, 2)

    else:
        # ── Added to queue, wait for opponent ────────────────────────
        entry = {**user, "ws": websocket, "game_id": None}
        matchmaking_queue.append(entry)
        await websocket.send_json({"type": "queued"})

        try:
            while entry["game_id"] is None:
                try:
                    raw  = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
                    data = json.loads(raw)
                    if data.get("type") == "cancel":
                        matchmaking_queue[:] = [q for q in matchmaking_queue if q["id"] != user["id"]]
                        await websocket.send_json({"type": "cancelled"})
                        return
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            matchmaking_queue[:] = [q for q in matchmaking_queue if q["id"] != user["id"]]
            return

        gid  = entry["game_id"]
        game = active_games.get(gid)
        if not game:
            return

        await _relay(gid, websocket, 1)
