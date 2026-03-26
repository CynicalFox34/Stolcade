"""
Stolcade game engine — pure Python, no dependencies.
Faithfully mirrors the JS implementation in index.html.
"""

import numpy as np
from copy import deepcopy

ROWS, COLS, MID_COL = 20, 11, 5
P1_GOAL, P2_GOAL, WIN_COUNT = 19, 0, 3
DIRS = [(-1,0),(1,0),(0,-1),(0,1)]

# Blocked cells at the center border (rows 9-10, cols 2,3,7,8)
BLOCKED = frozenset(
    (r, c) for r in (9, 10) for c in (2, 3, 7, 8)
)
def is_blocked(r, c): return (r, c) in BLOCKED

# ── Piece ────────────────────────────────────────────────────
class Piece:
    __slots__ = ('player','is_veylant','has_crossed','has_ever_crossed')
    def __init__(self, player, is_veylant=False, has_crossed=False, has_ever_crossed=False):
        self.player           = player
        self.is_veylant       = is_veylant
        self.has_crossed      = has_crossed
        self.has_ever_crossed = has_ever_crossed
    def clone(self):
        return Piece(self.player, self.is_veylant, self.has_crossed, self.has_ever_crossed)
    def __repr__(self):
        t = 'V' if self.is_veylant else ('x' if self.has_crossed else 'o')
        return f'P{self.player}{t}'

# ── Board state ──────────────────────────────────────────────
def initial_state():
    board = [[None]*COLS for _ in range(ROWS)]
    for r in range(ROWS):
        pl = 1 if r < 10 else 2
        is_v = (pl == 1 and r == 0) or (pl == 2 and r == 19)
        board[r][MID_COL] = Piece(pl, is_v)
    return board

def clone_board(board):
    return [[p.clone() if p else None for p in row] for row in board]

def count_at_goal(board, player):
    row = P1_GOAL if player == 1 else P2_GOAL
    return sum(1 for c in range(COLS) if board[row][c] and board[row][c].player == player)

def tiebreak_winner(board):
    """
    Tiebreaker when move limit is hit (no outright winner).
    Uses total progress of ALL pieces toward the opponent's goal.
    Higher total progress = winning player.
    """
    def total_progress(player):
        goal = P1_GOAL if player == 1 else P2_GOAL
        total = 0
        for r in range(ROWS):
            for c in range(COLS):
                p = board[r][c]
                if p and p.player == player:
                    # Progress = how close this piece is to the goal (max ROWS-1, min 0)
                    total += (ROWS - 1) - abs(r - goal)
                    if p.has_crossed: total += ROWS  # bonus for crossing
        return total

    s1, s2 = total_progress(1), total_progress(2)
    if s1 > s2: return 1
    if s2 > s1: return 2
    return 0  # genuine tie — neither side advanced more

def did_cross(player, from_r, to_r):
    return (from_r < 10 and to_r >= 10) if player == 1 else (from_r >= 10 and to_r < 10)

def blocked_by_border(piece, to_r):
    if not piece.has_crossed: return False
    if piece.player == 1 and to_r < 10:  return True
    if piece.player == 2 and to_r >= 10: return True
    return False

def veylant_steps(board, player):
    return 2 + count_at_goal(board, player)

# ── Move generation ──────────────────────────────────────────
def moves_for_regular(board, r, c):
    piece = board[r][c]
    if piece.player == 1 and r == P1_GOAL: return []
    if piece.player == 2 and r == P2_GOAL: return []

    out = []
    for dr, dc in DIRS:
        nr, nc = r+dr, c+dc
        push_path = [(r,c)]
        can_push = True
        sideways = (dr == 0)

        while True:
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                can_push = False; break
            if is_blocked(nr, nc):
                can_push = False; break
            target = board[nr][nc]
            if target is None:
                if blocked_by_border(piece, nr): can_push = False
                break
            if target.player != piece.player:
                # Enemy piece — any direction except mid column
                if nc == MID_COL: can_push = False; break
                push_strength = len(push_path)
                scan_r, scan_c = nr, nc
                enemy_chain_len = 0
                chain_ok = True
                while True:
                    enemy_chain_len += 1
                    next_r, next_c = scan_r + dr, scan_c + dc
                    if not (0 <= next_r < ROWS and 0 <= next_c < COLS): chain_ok = False; break
                    if is_blocked(next_r, next_c): chain_ok = False; break
                    s = board[next_r][next_c]
                    if s is None: break
                    if s.player == piece.player: chain_ok = False; break
                    if s.player != target.player: chain_ok = False; break
                    scan_r, scan_c = next_r, next_c
                if chain_ok and enemy_chain_len <= 1 and push_strength > enemy_chain_len:
                    out.append({'isEnemyPush': True, 'target': (nr, nc), 'dr': dr, 'dc': dc})
                can_push = False; break
            # ally — in own territory, no backward pushes
            own_territory = (piece.player == 1 and nr < 10) or (piece.player == 2 and nr >= 10)
            is_forward    = (piece.player == 1 and dr == 1) or (piece.player == 2 and dr == -1)
            if own_territory and not is_forward and not sideways:
                can_push = False; break
            push_path.append((nr,nc))
            nr += dr; nc += dc

        if can_push:
            out.append({'path': push_path, 'target': (nr,nc), 'dr': dr, 'dc': dc})
    return out

def moves_for_veylant(board, r, c):
    piece = board[r][c]
    if piece.player == 1 and r == P1_GOAL: return []
    if piece.player == 2 and r == P2_GOAL: return []

    steps = veylant_steps(board, piece.player)
    reachable = {}   # target_key -> moveobj
    visited = {}     # veylant_pos_key -> min_steps_used

    def simulate_push(b, sr, sc, dr, dc, er, ec):
        nb = clone_board(b)
        pr, pc = er-dr, ec-dc
        empty_r, empty_c = er, ec
        loops = 0
        while loops < 30:
            loops += 1
            if not (0 <= pr < ROWS and 0 <= pc < COLS): break
            p = nb[pr][pc]
            if p is None: break
            nb[empty_r][empty_c] = p
            if pr == sr and pc == sc: break
            empty_r, empty_c = pr, pc
            pr -= dr; pc -= dc
        nb[sr][sc] = None
        return nb

    queue = [(r, c, 0, [], board)]
    visited[r*COLS+c] = 0

    while queue:
        cr, cc, used, path_so_far, cur_board = queue.pop(0)
        if used >= steps: continue

        for dr, dc in DIRS:
            nr, nc = cr+dr, cc+dc
            can_push = True
            sideways = (dr == 0)
            push_count = 0
            hit_enemy = False
            first_enemy_r = -1
            first_enemy_c = -1

            while True:
                if not (0 <= nr < ROWS and 0 <= nc < COLS):
                    can_push = False; break
                if is_blocked(nr, nc):
                    can_push = False; break
                target = cur_board[nr][nc]
                if target is None:
                    if not hit_enemy and blocked_by_border(piece, nr): can_push = False
                    break
                if target.player != piece.player:
                    if not hit_enemy:
                        # First enemy — validate push using original board for push strength
                        if nc == MID_COL: can_push = False; break
                        bs_push_count = 0
                        bs_r, bs_c = cr + dr, cc + dc
                        while 0 <= bs_r < ROWS and 0 <= bs_c < COLS:
                            if bs_r == r and bs_c == c: break
                            bsp = board[bs_r][bs_c]
                            if bsp is None or bsp.player != piece.player: break
                            bs_push_count += 1
                            bs_r += dr; bs_c += dc
                        push_strength = bs_push_count + 1
                        scan_r, scan_c = nr, nc
                        enemy_chain_len = 0
                        chain_ok = True
                        while True:
                            enemy_chain_len += 1
                            next_r, next_c = scan_r + dr, scan_c + dc
                            if not (0 <= next_r < ROWS and 0 <= next_c < COLS): chain_ok = False; break
                            if is_blocked(next_r, next_c): chain_ok = False; break
                            s = cur_board[next_r][next_c]
                            if s is None: break
                            if s.player == piece.player: chain_ok = False; break
                            if s.player != target.player: chain_ok = False; break
                            scan_r, scan_c = next_r, next_c
                        if not (chain_ok and enemy_chain_len <= 1 and push_strength > enemy_chain_len):
                            can_push = False; break
                        hit_enemy = True
                        first_enemy_r, first_enemy_c = nr, nc
                    nr += dr; nc += dc
                    continue
                # Ally piece — in own territory, no backward pushes
                if hit_enemy: can_push = False; break
                own_territory = (piece.player==1 and nr<10) or (piece.player==2 and nr>=10)
                is_forward    = (piece.player==1 and dr==1) or (piece.player==2 and dr==-1)
                if own_territory and not is_forward and not sideways:
                    can_push = False; break
                push_count += 1
                nr += dr; nc += dc

            if not can_push: continue

            next_used = used + 1
            if next_used > steps: continue

            v_new_r, v_new_c = cr+dr, cc+dc
            new_path = path_so_far + [{'dr':dr,'dc':dc,'er':nr,'ec':nc,
                                       'isEnemyPush':hit_enemy,
                                       'enemyTarget':(first_enemy_r,first_enemy_c) if hit_enemy else None}]

            if hit_enemy:
                # Veylant lands on first enemy's current square
                land_key = v_new_r*COLS+v_new_c
                if land_key != r*COLS+c and land_key not in reachable:
                    reachable[land_key] = {'target':(v_new_r,v_new_c),'is_veylant_multi':True,'path_sequence':new_path}
                
                # Continue BFS from landing position with pushed board
                vkey = v_new_r*COLS+v_new_c
                if vkey not in visited or visited[vkey] >= next_used:
                    visited[vkey] = next_used
                    next_board = simulate_push(cur_board, cr, cc, dr, dc, nr, nc)
                    just_crossed = did_cross(piece.player, cr, v_new_r)
                    if not just_crossed:
                        queue.append((v_new_r, v_new_c, next_used, new_path, next_board))
            else:
                key = nr*COLS+nc
                if key != r*COLS+c and key not in reachable:
                    reachable[key] = {'target':(nr,nc),'is_veylant_multi':True,'path_sequence':new_path}

                just_crossed = did_cross(piece.player, cr, v_new_r)
                vkey = v_new_r*COLS+v_new_c
                if not just_crossed and (vkey not in visited or visited[vkey] >= next_used):
                    visited[vkey] = next_used
                    if push_count > 0:
                        next_board = simulate_push(cur_board, cr, cc, dr, dc, nr, nc)
                    else:
                        next_board = clone_board(cur_board)
                        next_board[v_new_r][v_new_c] = next_board[cr][cc]
                        next_board[cr][cc] = None
                    queue.append((v_new_r, v_new_c, next_used, new_path, next_board))

    return list(reachable.values())

def get_valid_moves(board, r, c):
    piece = board[r][c]
    if piece is None: return []
    return moves_for_veylant(board, r, c) if piece.is_veylant else moves_for_regular(board, r, c)

def get_all_moves(board, player):
    moves = []
    for r in range(ROWS):
        for c in range(COLS):
            if board[r][c] and board[r][c].player == player:
                for m in get_valid_moves(board, r, c):
                    moves.append((r, c, m))
    return moves

# ── Move execution ────────────────────────────────────────────
def apply_move(board, from_r, from_c, move, player):
    """Returns (new_board, bonus_earned). Does NOT mutate input."""
    b = clone_board(board)
    bonus_earned = [0, 0, 0]  # indexed by player

    def push_line(sr, sc, dr, dc, er, ec):
        nonlocal bonus_earned
        pr, pc = er-dr, ec-dc
        empty_r, empty_c = er, ec
        loops = 0
        while loops < 30:
            loops += 1
            if not (0 <= pr < ROWS and 0 <= pc < COLS): break
            p = b[pr][pc]
            if p is None: break
            b[empty_r][empty_c] = p
            # Any piece crossing into enemy territory earns its owner a bonus (first time only)
            if not p.has_ever_crossed and did_cross(p.player, pr, empty_r):
                p.has_crossed = True
                p.has_ever_crossed = True
                p.is_veylant = False
                bonus_earned[p.player] += 1
            # Reset has_crossed if a piece is pushed back to its own half
            if p.has_crossed:
                if p.player == 1 and empty_r < 10:  p.has_crossed = False
                if p.player == 2 and empty_r >= 10: p.has_crossed = False
            if pr == sr and pc == sc: break
            empty_r, empty_c = pr, pc
            pr -= dr; pc -= dc
        b[sr][sc] = None

    def find_empty_past_enemies(start_r, start_c, dr, dc):
        """Scan forward from an enemy position to find the empty cell beyond the chain."""
        scan_r, scan_c = start_r, start_c
        while True:
            next_r, next_c = scan_r + dr, scan_c + dc
            if not (0 <= next_r < ROWS and 0 <= next_c < COLS): break
            if is_blocked(next_r, next_c): break
            s = b[next_r][next_c]
            if s is None or s.player == player: break
            scan_r, scan_c = next_r, next_c
        return scan_r + dr, scan_c + dc

    if 'path' in move:
        push_line(from_r, from_c, move['dr'], move['dc'], move['target'][0], move['target'][1])
    elif move.get('isEnemyPush'):
        er, ec = find_empty_past_enemies(move['target'][0], move['target'][1], move['dr'], move['dc'])
        push_line(from_r, from_c, move['dr'], move['dc'], er, ec)
    elif move.get('is_veylant_multi'):
        cr, cc = from_r, from_c
        for step in move['path_sequence']:
            if step.get('isEnemyPush'):
                et = step['enemyTarget']
                er, ec = find_empty_past_enemies(et[0], et[1], step['dr'], step['dc'])
                push_line(cr, cc, step['dr'], step['dc'], er, ec)
                cr, cc = et[0], et[1]  # veylant lands on first enemy's original square
            else:
                push_line(cr, cc, step['dr'], step['dc'], step['er'], step['ec'])
                cr += step['dr']; cc += step['dc']

    return b, bonus_earned

# ── Game state class ─────────────────────────────────────────
class GameState:
    def __init__(self):
        self.board          = initial_state()
        self.current_player = 1
        self.actions_left   = 1
        self.bonus_pending  = [0, 0, 0]  # index 0 unused; counts extra actions earned
        self.game_over      = False
        self.winner         = None
        self.move_count     = 0

    def clone(self):
        gs = GameState.__new__(GameState)
        gs.board          = clone_board(self.board)
        gs.current_player = self.current_player
        gs.actions_left   = self.actions_left
        gs.bonus_pending  = self.bonus_pending[:]
        gs.game_over      = self.game_over
        gs.winner         = self.winner
        gs.move_count     = self.move_count
        return gs

    def legal_moves(self):
        if self.game_over: return []
        return get_all_moves(self.board, self.current_player)

    def apply(self, from_r, from_c, move):
        """Returns a new GameState with the move applied."""
        gs = self.clone()
        new_board, bonus_earned = apply_move(gs.board, from_r, from_c, move, gs.current_player)
        gs.board = new_board
        gs.move_count += 1

        nb = gs.bonus_pending[:]
        nb[1] += bonus_earned[1]
        nb[2] += bonus_earned[2]

        # Win check
        if count_at_goal(gs.board, gs.current_player) >= WIN_COUNT:
            gs.game_over = True
            gs.winner    = gs.current_player
            return gs

        # Advance turn
        if gs.actions_left > 1:
            gs.actions_left -= 1
        else:
            next_p = 3 - gs.current_player  # 1->2, 2->1
            gs.actions_left = 1 + nb[next_p]
            nb[next_p] = 0
            gs.current_player = next_p

        gs.bonus_pending = nb
        return gs

    def is_terminal(self):
        return self.game_over

    def to_tensor(self):
        """
        Canonical (player-relative) encoding — always from current player's POV.
        For P2 the board is flipped vertically so both players see their pieces
        advancing toward row 19 (goal) from row 0 (home).

        Channels:
          0 - current player's pieces
          1 - opponent's pieces
          2 - current player's crossed pieces
          3 - opponent's crossed pieces
          4 - current player's veylant
          5 - opponent's veylant
          6 - current player's goal row (row 19 in canonical view)
          7 - opponent's goal row (row 0 in canonical view)
          8 - all 1s (always current player's view)
          9 - center line (rows 9-10)
         10 - current player's bonus actions pending (normalized by /3, capped 1)
         11 - opponent's bonus actions pending (normalized by /3, capped 1)
        """
        cp   = self.current_player
        op   = 3 - cp
        flip = (cp == 2)   # flip board rows for P2 so goal is always row 19

        t = np.zeros((12, ROWS, COLS), dtype=np.float32)
        for r in range(ROWS):
            cr = (ROWS - 1 - r) if flip else r   # canonical row
            for c in range(COLS):
                p = self.board[r][c]
                if p:
                    ch = 0 if p.player == cp else 1
                    t[ch,   cr, c] = 1.0
                    if p.has_crossed:
                        t[ch+2, cr, c] = 1.0
                    if p.is_veylant:
                        t[ch+4, cr, c] = 1.0
        t[6, P1_GOAL, :] = 1.0   # row 19 — current player's goal in canonical view
        t[7, P2_GOAL, :] = 1.0   # row  0 — opponent's goal in canonical view
        t[8, :, :]       = 1.0   # always current player's view
        t[9, 9,  :]      = 1.0
        t[9, 10, :]      = 1.0
        t[10, :, :]      = min(self.bonus_pending[cp], 3) / 3.0
        t[11, :, :]      = min(self.bonus_pending[op], 3) / 3.0
        return t

    def get_hash(self):
        """Returns a stable hash of the board state for repetition checks."""
        items = []
        for r in range(ROWS):
            for c in range(COLS):
                p = self.board[r][c]
                if p:
                    items.append((r, c, p.player, p.is_veylant, p.has_crossed))
        return hash((tuple(items), self.current_player, self.actions_left))

    def __repr__(self):
        lines = []
        for r in range(ROWS):
            row_str = ''
            for c in range(COLS):
                p = self.board[r][c]
                if p is None:
                    row_str += '. '
                else:
                    row_str += str(p) + ' '
            lines.append(f'{r:2d} {row_str}')
        return '\n'.join(lines)
