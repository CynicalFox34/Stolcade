#!/usr/bin/env python3
"""
Stolcade bot API — serves /api/bot_move requests from the JS frontend.
Uses the Python minimax engine (mirrors the JS Hard/Medium/Easy bots exactly).
"""
import os, sys, json

ENGINE_DIR = os.path.dirname(os.path.abspath(__file__))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

from game import GameState, Piece, ROWS, COLS, MID_COL
from minimax import minimax_move

DEPTH = {'Easy Bot': 1, 'Medium Bot': 2, 'Hard Bot': 3,
         'easy': 1, 'medium': 2, 'hard': 3,
         'basic': 1, 'advanced': 3}


def board_from_json(board_json):
    """
    Convert the JS board encoding to a GameState.
    JS sends each cell as:  1=P1 piece, 2=P1 veylant, -1=P2 piece, -2=P2 veylant, 0=empty
    """
    gs = GameState.__new__(GameState)
    gs.board = [[None] * COLS for _ in range(ROWS)]
    gs.current_player = 1   # caller sets the player before calling get_bot_move
    gs.actions_left   = 1
    gs.bonus_pending  = [False, False, False]
    gs.game_over      = False
    gs.winner         = None
    gs.move_count     = 0

    for r in range(ROWS):
        for c in range(COLS):
            v = board_json[r][c]
            if v == 0: continue
            player    = 1 if v > 0 else 2
            is_veylant = abs(v) == 2
            # Infer has_crossed from position
            has_crossed = (player == 1 and r >= 10) or (player == 2 and r < 10)
            gs.board[r][c] = Piece(player, is_veylant, has_crossed)

    return gs


def get_bot_move(board_json, level='Hard Bot', player=2, pos_history=None):
    """
    board_json : 20x11 list of ints (JS encoding)
    level      : difficulty string
    player     : which player the bot is (1 or 2)
    Returns    : {'from': [r,c], 'to': [r,c]} or {'error': str}
    """
    try:
        if isinstance(board_json, str):
            board_json = json.loads(board_json)

        gs = board_from_json(board_json)
        gs.current_player = player

        depth = DEPTH.get(level, 3)
        move  = minimax_move(gs, depth=depth, pos_history=pos_history or [])

        if move is None:
            return {'error': 'No legal moves'}

        fr, fc, m = move
        tr, tc = m['target']
        return {'from': [fr, fc], 'to': [tr, tc], 'level': level}

    except Exception as e:
        return {'error': str(e)}
