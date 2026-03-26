"""
WebSocket-based online matchmaking and game relay.

Flow:
  1. Client connects to /ws/play?token=<jwt>
  2. Server validates token, adds player to matchmaking queue
  3. When two players are queued, they are paired into a game
  4. Each player's connection relays their moves to the opponent
  5. On game_over, ELO is updated for both players
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from jose import JWTError, jwt
import asyncio, json

from ..database import SessionLocal
from ..models import User, Match
from ..auth import SECRET_KEY, ALGORITHM

router = APIRouter()

matchmaking_queue: list = []   # [{id, username, elo, ws, game_id}]
active_games: dict     = {}    # gid -> {p1: {...}, p2: {...}}
_counter               = 0


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
    """Update ELO for both players and save match record. Pops game to prevent double-update."""
    game = active_games.pop(gid, None)
    if not game:
        return
    db = SessionLocal()
    try:
        p1 = db.query(User).filter(User.id == game["p1"]["id"]).first()
        p2 = db.query(User).filter(User.id == game["p2"]["id"]).first()
        if not (p1 and p2):
            return
        K    = 32
        exp1 = 1 / (1 + 10 ** ((p2.elo - p1.elo) / 400))
        exp2 = 1 - exp1

        p1_elo_before = round(p1.elo)
        p2_elo_before = round(p2.elo)

        if winner == 1:
            s1, s2 = 1.0, 0.0; p1.wins   += 1; p2.losses += 1; r1, r2 = 'win',  'loss'
        elif winner == 2:
            s1, s2 = 0.0, 1.0; p1.losses += 1; p2.wins   += 1; r1, r2 = 'loss', 'win'
        else:
            s1, s2 = 0.5, 0.5; p1.draws  += 1; p2.draws  += 1; r1, r2 = 'draw', 'draw'

        p1.elo = max(100, round(p1.elo + K * (s1 - exp1)))
        p2.elo = max(100, round(p2.elo + K * (s2 - exp2)))

        match = Match(
            player1_id    = p1.id,
            player2_id    = p2.id,
            result_p1     = r1,
            elo_p1_before = p1_elo_before,
            elo_p1_after  = round(p1.elo),
            elo_p2_before = p2_elo_before,
            elo_p2_after  = round(p2.elo),
            is_bot        = False,
        )
        db.add(match)
        db.commit()
    finally:
        db.close()


async def _relay(gid: str, my_ws: WebSocket, opp_ws: WebSocket):
    """Relay messages from my_ws to opp_ws until game over or disconnect."""
    try:
        while True:
            raw  = await my_ws.receive_text()
            data = json.loads(raw)
            t    = data.get("type")

            if t == "move":
                try:
                    await opp_ws.send_json(data)
                except Exception:
                    pass

            elif t == "game_over":
                try:
                    await opp_ws.send_json(data)
                except Exception:
                    pass
                await _update_elo(gid, data.get("winner"))
                return

    except WebSocketDisconnect:
        try:
            await opp_ws.send_json({"type": "opponent_left"})
        except Exception:
            pass
    finally:
        active_games.pop(gid, None)


@router.websocket("/ws/play")
async def ws_play(websocket: WebSocket, token: str = Query(...)):
    await websocket.accept()

    user = await _get_user(token)
    if not user:
        await websocket.send_json({"type": "error", "msg": "Unauthorized"})
        await websocket.close()
        return

    global _counter, matchmaking_queue

    # Remove any stale entry for this user (reconnect case)
    matchmaking_queue = [q for q in matchmaking_queue if q["id"] != user["id"]]

    if matchmaking_queue:
        # ── Matched immediately ──────────────────────────────────────────
        opp = matchmaking_queue.pop(0)
        _counter += 1
        gid = f"g{_counter}"

        active_games[gid] = {
            "p1": {**opp,  "ws": opp["ws"]},
            "p2": {**user, "ws": websocket},
        }
        opp["game_id"] = gid   # signals opp's queue-wait loop to exit

        await opp["ws"].send_json({
            "type": "matched", "game_id": gid,
            "player": 1, "opponent": user["username"], "opponent_elo": user["elo"],
        })
        await websocket.send_json({
            "type": "matched", "game_id": gid,
            "player": 2, "opponent": opp["username"], "opponent_elo": opp["elo"],
        })

        # This handler relays p2 → p1
        await _relay(gid, websocket, opp["ws"])

    else:
        # ── Added to queue, wait for opponent ────────────────────────────
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
                    pass   # keep waiting
        except WebSocketDisconnect:
            matchmaking_queue[:] = [q for q in matchmaking_queue if q["id"] != user["id"]]
            return

        # Now matched — relay p1 → p2
        gid  = entry["game_id"]
        game = active_games.get(gid)
        if not game:
            return
        await _relay(gid, websocket, game["p2"]["ws"])
