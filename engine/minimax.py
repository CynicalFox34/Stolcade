"""
Stolcade minimax bot — Python port of the JavaScript Hard/Medium/Easy bots.
Mirrors evalBoard + minimax + alpha-beta from index.html exactly.

Depths: Easy=1, Medium=3, Hard=5
"""
import random
import numpy as np
from game import (
    ROWS, COLS, P1_GOAL, P2_GOAL, WIN_COUNT,
    get_all_moves, apply_move, count_at_goal, did_cross, clone_board
)


def eval_board(board, bonus_pending, max_player):
    min_player = 3 - max_player
    if count_at_goal(board, max_player) >= WIN_COUNT: return  100000
    if count_at_goal(board, min_player) >= WIN_COUNT: return -100000
    score = 0
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r][c]
            if p is None: continue
            sign = 1 if p.player == max_player else -1
            goal = P1_GOAL if p.player == 1 else P2_GOAL
            dist = abs(r - goal)
            score += sign * (ROWS - 1 - dist) * 3
            if p.has_crossed: score += sign * 8
            if dist == 0:     score += sign * 40
    score += 6 * bonus_pending[max_player]
    score -= 6 * bonus_pending[min_player]
    return score


def minimax(board, depth, alpha, beta, is_max, player, actions_left,
            bonus_pending, max_player, pos_history):
    min_player = 3 - max_player
    if count_at_goal(board, max_player) >= WIN_COUNT: return  100000
    if count_at_goal(board, min_player) >= WIN_COUNT: return -100000
    if depth == 0:
        return eval_board(board, bonus_pending, max_player)

    moves = get_all_moves(board, player)

    if not moves:
        # No moves — pass turn
        next_p = 3 - player
        nb = bonus_pending[:]
        na = 1 + nb[next_p]
        nb[next_p] = 0
        return minimax(board, depth - 1, alpha, beta, not is_max,
                       next_p, na, nb, max_player, pos_history)

    # Move ordering: backline first, crossing second (same as JS)
    goal = P1_GOAL if player == 1 else P2_GOAL
    def rank(move):
        fr, fc, m = move
        tr = m['target'][0]
        if tr == goal:               return 2
        if did_cross(player, fr, tr): return 1
        return 0
    moves.sort(key=rank, reverse=True)

    best = -float('inf') if is_max else float('inf')
    for fr, fc, m in moves:
        new_board, bonus_earned = apply_move(board, fr, fc, m, player)
        nb = bonus_pending[:]
        nb[1] += bonus_earned[1]
        nb[2] += bonus_earned[2]

        if actions_left > 1:
            next_player  = player
            next_actions = actions_left - 1
            next_is_max  = is_max
        else:
            next_player  = 3 - player
            next_actions = 1 + nb[next_player]
            nb[next_player] = 0
            next_is_max  = not is_max

        val = minimax(new_board, depth - 1, alpha, beta, next_is_max,
                      next_player, next_actions, nb, max_player, pos_history)

        # Repetition penalty: penalise revisiting recent positions
        board_key = _board_key(new_board)
        if pos_history.count(board_key) >= 1:
            val = val - 300 if is_max else val + 300

        if is_max:
            best  = max(best, val)
            alpha = max(alpha, val)
        else:
            best = min(best, val)
            beta = min(beta, val)
        if beta <= alpha:
            break

    return best


def minimax_move(gs, depth=3, pos_history=None, temperature=0.0, forbidden_positions=None):
    """
    Return (fr, fc, move_obj) for the best move at the given depth.
    depth: 1=Easy, 3=Medium, 5=Hard
    pos_history: list of recent board keys (for repetition penalty)
    temperature: if > 0, sample from softmax of normalized move values instead of argmax
    forbidden_positions: set of board keys to exclude (forces new positions)
    """
    if pos_history is None:
        pos_history = []

    player = gs.current_player
    moves  = gs.legal_moves()
    if not moves: return None

    bonus_pending = gs.bonus_pending[:]

    # Move ordering at root
    goal = P1_GOAL if player == 1 else P2_GOAL
    def rank(move):
        fr, fc, m = move
        tr = m['target'][0]
        if tr == goal:               return 2
        if did_cross(player, fr, tr): return 1
        return 0
    moves.sort(key=rank, reverse=True)

    # Filter out moves leading to already-seen positions (if any fresh moves exist)
    if forbidden_positions:
        fresh = [mv for mv in moves
                 if _board_key(apply_move(gs.board, mv[0], mv[1], mv[2], player)[0])
                 not in forbidden_positions]
        if fresh:
            moves = fresh

    best_move = None
    best_val  = -float('inf')
    alpha     = -float('inf')
    beta      = float('inf')
    all_vals  = []

    for fr, fc, m in moves:
        new_board, bonus_earned = apply_move(gs.board, fr, fc, m, player)
        nb = bonus_pending[:]
        nb[1] += bonus_earned[1]
        nb[2] += bonus_earned[2]

        if gs.actions_left > 1:
            next_player  = player
            next_actions = gs.actions_left - 1
            next_is_max  = True
        else:
            next_player  = 3 - player
            next_actions = 1 + nb[next_player]
            nb[next_player] = 0
            next_is_max  = False

        val = minimax(new_board, depth - 1, alpha, beta, next_is_max,
                      next_player, next_actions, nb, player, pos_history)
        all_vals.append(val)

        if val > best_val:
            best_val  = val
            best_move = (fr, fc, m)
        alpha = max(alpha, val)

    # Temperature sampling: normalize values to [-1,1] then softmax
    if temperature > 0 and len(moves) > 1:
        vals = np.array(all_vals, dtype=np.float32)
        vmin, vmax = vals.min(), vals.max()
        if vmax > vmin:
            vals = 2.0 * (vals - vmin) / (vmax - vmin) - 1.0  # normalize to [-1, 1]
        vals = vals / temperature
        vals -= vals.max()
        probs = np.exp(vals)
        probs /= probs.sum()
        idx = int(np.random.choice(len(moves), p=probs))
        return moves[idx]

    return best_move


def _board_key(board):
    """Compact hash of piece positions (same logic as JS boardHash)."""
    return ''.join(
        str(board[r][c].player) if board[r][c] else '0'
        for r in range(ROWS) for c in range(COLS)
    )
