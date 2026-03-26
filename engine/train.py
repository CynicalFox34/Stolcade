"""
Stolcade AlphaZero-style MCTS self-play trainer.

Move selection uses batched MCTS (virtual loss trick — 16 parallel sims per
GPU batch).  Labels are pure win/loss/tiebreak — no reward shaping.

Run:
  cd /Users/lincoln/Documents/stolcade/engine
  python3 train.py

Outputs:
  checkpoints/best.pt    — strongest model so far
  checkpoints/latest.pt  — most recent weights
"""

import os, time, random, sys, json, shutil
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

from game import GameState, did_cross, P1_GOAL, P2_GOAL, ROWS, COLS, tiebreak_winner
from model import StokcadeNet
from minimax import minimax_move, _board_key

# ── Config ───────────────────────────────────────────────────
DEVICE          = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
CHANNELS        = 128
RES_BLOCKS      = 12
LR              = 3e-4
BATCH_SIZE      = 256
REPLAY_SIZE     = 50_000
GAMES_PER_ITER  = 5       # MCTS games are higher quality than depth-1
TRAIN_STEPS     = 80
EVAL_EVERY      = 5
EVAL_GAMES      = 8
WIN_THRESHOLD   = 0.55
MAX_MOVES       = 90
WARMUP_ITERS    = 3       # Greedy-game warm-up iterations before MCTS begins
ELO_BASE        = 1000.0
ELO_K           = 32
CHECKPOINT_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
HEARTBEAT_FILE  = os.path.join(CHECKPOINT_DIR, 'train.heartbeat')
LOG_FILE        = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training.log')

# MCTS params
MCTS_SIMS        = 200     # Simulations during self-play
MCTS_EVAL_SIMS   = 200     # Much deeper search for benchmarks
MCTS_BATCH       = 16
C_PUCT           = 2.0     # Exploration weight
DIRICHLET_ALPHA  = 0.3
DIRICHLET_EPS    = 0.25

# Strategic Biases
REPETITION_PENALTY = 0.4    # Additive score penalty if move repeats a recent ancestor state
FORWARD_BIAS       = 0.15   # Bonus for moves advancing toward goal; stall detection uses repetition not progress

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

_log_f = open(LOG_FILE, 'a')
class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, d):
        for s in self.streams: s.write(d); s.flush()
    def flush(self):
        for s in self.streams: s.flush()
sys.stdout = _Tee(sys.__stdout__, _log_f)
sys.stderr = _Tee(sys.__stderr__, _log_f)

# ── Greedy baseline move ──────────────────────────────────────
def greedy_move(gs):
    """
    Deterministic forward-progress heuristic for warmup games.
    Always advances the piece closest to the goal; prefers moves that
    cross the midline or reach the goal row; breaks ties by distance gained.
    No random noise — guarantees monotone progress to produce decisive outcomes.
    """
    moves = gs.legal_moves()
    if not moves: return None
    player = gs.current_player
    goal   = P1_GOAL if player == 1 else P2_GOAL
    best, best_score = None, -1e9
    for (fr, fc, m) in moves:
        tr   = m['target'][0]
        dist_before = abs(fr - goal)
        dist_after  = abs(tr - goal)
        gain = dist_before - dist_after          # positive = moving toward goal
        score = gain * 100                       # prioritise pieces that advance most
        score -= dist_after * 10                 # among equal-gain, prefer most-advanced piece
        if tr == goal:               score += 2000  # reaching goal is decisive
        if did_cross(player, fr, tr): score += 500  # crossing midline is huge
        p = gs.board[fr][fc]
        if p and p.has_crossed:      score += 50
        if score > best_score: best_score = score; best = (fr, fc, m)
    return best

def total_progress(board):
    """Stall detection: sum of all pieces' advancement (crossing included)."""
    total = 0
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r][c]
            if p:
                goal = P1_GOAL if p.player == 1 else P2_GOAL
                total += (ROWS - 1) - abs(r - goal)
                if p.has_crossed: total += ROWS
    return total

# ── MCTS ─────────────────────────────────────────────────────
class MCTSNode:
    """
    One node in the MCTS tree, representing a game state reached by some action.

    Q-value convention (negamax):
        node.Q  is always from node.gs.current_player's perspective.
    This means select_child must negate Q when the child belongs to the opponent.
    Virtual loss temporarily deflates Q to discourage parallel sims from
    piling onto the same path.

    Game states are computed LAZILY: child nodes store only the move needed to
    reach them. The game state is computed (and cached) the first time .gs is
    accessed, which happens when the child is actually traversed during selection.
    This avoids cloning 20+ game states per expansion when only 1-2 are visited.
    """
    __slots__ = ['_gs', '_move', 'same_player', 'moves', 'prior', 'parent',
                 'children', 'N', 'W', 'virtual_loss', 'child_biases']

    def __init__(self, gs=None, prior=1.0, parent=None, move=None, same_player=True):
        self._gs          = gs
        self._move        = move    # move to apply to parent.gs to get this state
        self.same_player  = same_player  # child's current_player == parent's current_player
        self.moves        = None   # lazily populated on first expand
        self.prior        = prior
        self.parent       = parent
        self.children     = []     # list[MCTSNode], same order as self.moves
        self.N            = 0
        self.W            = 0.0
        self.virtual_loss = 0
        self.child_biases = []   # Pre-calculated forward bonuses (all 0 when FORWARD_BIAS=0)

    @property
    def gs(self):
        """Lazily compute and cache the game state by applying _move to parent.gs."""
        if self._gs is None:
            self._gs = self.parent.gs.apply(*self._move)
        return self._gs

    @property
    def Q(self):
        return self.W / self.N if self.N > 0 else 0.0

    def is_leaf(self):
        return len(self.children) == 0

    def select_child(self, c_puct):
        """PUCT selection — returns best child from the current player's perspective.

        Virtual loss is handled as an explicit additive penalty rather than being
        folded into Q. The old approach (Q = (W-VL)/(N+VL)) correctly discouraged
        re-selection for same-player nodes but accidentally *encouraged* it for
        cross-player nodes (negating a negative Q made it look positive). The fix:
        compute raw Q from parent's perspective, then subtract a player-agnostic
        virtual loss penalty that always discourages re-selection.
        """
        parent_n   = self.N
        best_score = -1e9
        best_child = None
        for i, child in enumerate(self.children):
            # Raw Q from parent's perspective (no virtual loss distortion)
            raw_q = child.W / child.N if child.N > 0 else 0.0
            q     = raw_q if child.same_player else -raw_q

            # Exploration bonus (VL in denominator reduces U for busy nodes)
            u = c_puct * child.prior * (parent_n ** 0.5) / (1 + child.N + child.virtual_loss)

            # Virtual loss penalty: always discourages regardless of player perspective.
            # Magnitude = 1.0 per VL when N=0 (matches the worst-case Q range of [-1,1]).
            vl_penalty = child.virtual_loss / max(1, child.N)

            score = q + u + self.child_biases[i] - vl_penalty
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def expand(self, policy_logits):
        """
        Create child nodes from policy logits (220-dim, unnormalized).
        Priors are softmax over legal-move target squares (canonical coords).
        Child game states are NOT computed here — they are lazily evaluated
        the first time each child is selected during simulation.
        """
        if self.moves is None:
            self.moves = self.gs.legal_moves()
        if not self.moves:
            return
        cp   = self.gs.current_player
        flip = (cp == 2)
        goal = P1_GOAL if cp == 1 else P2_GOAL

        idxs = [((ROWS-1-m[2]['target'][0]) if flip else m[2]['target'][0]) * 11
                + m[2]['target'][1] for m in self.moves]
        logits = np.array([policy_logits[i] for i in idxs], dtype=np.float64)
        logits -= logits.max()
        priors  = np.exp(logits)
        priors /= priors.sum()

        # same_player: True if this node's current_player keeps the turn (multi-action),
        # False if the turn passes to the other player after this move.
        child_same_player = (self.gs.actions_left > 1)

        for i, move in enumerate(self.moves):
            # Store the move only — game state computed lazily on first visit
            child = MCTSNode(prior=float(priors[i]), parent=self,
                             move=move, same_player=child_same_player)
            self.children.append(child)

            # Forward bias: bonus for moves advancing a piece toward goal
            fr, fc = move[0], move[1]
            tr, tc = move[2]['target']
            self.child_biases.append(FORWARD_BIAS if abs(tr-goal) < abs(fr-goal) else 0.0)


def _backup(path, leaf_value):
    """
    Backpropagate leaf_value through the path.

    path:       [root, …, leaf] (MCTSNode list)
    leaf_value: value from path[-1].gs.current_player's perspective.

    Walking *up* the path: whenever the player changes between adjacent nodes,
    negate the value so it's always from the current node's player's perspective.
    Virtual loss is undone at each node.
    """
    value = leaf_value
    for i in range(len(path) - 1, -1, -1):
        node              = path[i]
        node.N            += 1
        node.W            += value
        node.virtual_loss -= 1
        if i > 0 and not node.same_player:
            value = -value


def run_mcts(root_gs, net, n_sims=MCTS_SIMS, add_noise=True, temperature=1.0):
    """
    Run batched MCTS from root_gs.

    Returns (chosen_move, pi) where:
      chosen_move — the move selected (sampled by temperature from visit counts)
      pi          — 220-dim visit-count distribution (policy training target)

    Returns (None, None) if root_gs is already terminal or has no legal moves.
    """
    if root_gs.is_terminal():
        return None, None

    moves = root_gs.legal_moves()
    if not moves:
        return None, None

    root = MCTSNode(gs=root_gs)

    # ── Initialise root policy priors ───────────────────────
    state_t = torch.tensor(root_gs.to_tensor()[None], dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        p_out, _ = net(state_t)
    root_policy = p_out[0].cpu().numpy()

    if add_noise and len(moves) > 0:
        root_flip  = (root_gs.current_player == 2)
        legal_idxs = [((ROWS-1-m[2]['target'][0]) if root_flip else m[2]['target'][0]) * 11
                      + m[2]['target'][1] for m in moves]
        noise      = np.random.dirichlet([DIRICHLET_ALPHA] * len(moves))
        for j, idx in enumerate(legal_idxs):
            root_policy[idx] = ((1 - DIRICHLET_EPS) * root_policy[idx]
                                + DIRICHLET_EPS * noise[j])

    root.expand(root_policy)
    root.N = 1  # count the root's own evaluation as one visit

    # ── Batched simulation loop ──────────────────────────────
    sims_done = 0
    while sims_done < n_sims:
        batch_sz = min(MCTS_BATCH, n_sims - sims_done)

        # Selection: walk each sim down to a leaf, applying virtual loss
        leaves = []
        paths  = []
        for _ in range(batch_sz):
            node = root
            path = [node]
            node.virtual_loss += 1

            while not node.is_leaf() and not node.gs.is_terminal():
                child = node.select_child(C_PUCT)
                if child is None:
                    break
                child.virtual_loss += 1
                path.append(child)
                node = child

            leaves.append(node)
            paths.append(path)

        # Split leaves into terminal / non-terminal
        term_idx     = [i for i, lf in enumerate(leaves) if lf.gs.is_terminal()]
        non_term_idx = [i for i, lf in enumerate(leaves) if not lf.gs.is_terminal()]

        values = np.zeros(len(leaves))

        # Terminal leaf values.
        # apply() does NOT switch current_player after a win — winner stays as current_player.
        # The correct value to assign depends on same_player (the negamax convention):
        #   same_player=False → backup will negate this value → assign -1.0 → parent gets +1.0 ✓
        #   same_player=True  → backup will NOT negate     → assign +1.0 → parent gets +1.0 ✓
        # same_player=False covers normal (single-action) wins.
        # same_player=True covers multi-action wins (player had bonus actions and won mid-turn).
        for i in term_idx:
            lf = leaves[i]
            if lf.gs.winner == 0:
                values[i] = 0.0
            else:
                values[i] = 1.0 if lf.same_player else -1.0

        # Batch-evaluate non-terminal leaves and expand them
        if non_term_idx:
            nt_leaves  = [leaves[i] for i in non_term_idx]
            state_arr  = np.array([lf.gs.to_tensor() for lf in nt_leaves])
            st         = torch.tensor(state_arr, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                p_batch, v_batch = net(st)
            p_np = p_batch.cpu().numpy()
            v_np = v_batch.squeeze(-1).cpu().numpy()

            for j, i in enumerate(non_term_idx):
                values[i] = float(v_np[j])   # from leaf's current_player perspective
                leaves[i].expand(p_np[j])    # expand with policy priors

        # Backpropagation
        for i in range(len(leaves)):
            _backup(paths[i], values[i])

        sims_done += batch_sz

    # ── Extract visit-count policy and choose a move ─────────
    pi           = np.zeros(220, dtype=np.float32)
    child_visits = np.array([c.N for c in root.children], dtype=np.float32)
    total_visits = child_visits.sum()

    root_flip = (root_gs.current_player == 2)
    if total_visits > 0 and root.moves:
        # Record pi using canonical row coordinates
        for j, move in enumerate(root.moves):
            tr, tc  = move[2]['target']
            canon_r = (ROWS - 1 - tr) if root_flip else tr
            pi[canon_r * 11 + tc] += child_visits[j] / total_visits

        if temperature < 0.01:
            chosen_idx = int(np.argmax(child_visits))
        else:
            probs      = child_visits ** (1.0 / temperature)
            probs     /= probs.sum()
            chosen_idx = int(np.random.choice(len(root.moves), p=probs))

        return root.moves[chosen_idx], pi

    # Fallback: uniform random
    if root.moves:
        idx     = random.randrange(len(root.moves))
        tr, tc  = root.moves[idx][2]['target']
        canon_r = (ROWS - 1 - tr) if root_flip else tr
        pi[canon_r * 11 + tc] = 1.0
        return root.moves[idx], pi

    return None, None


# ── Play one full game ────────────────────────────────────────
def shaped_outcome(winner, player, board, outright=False):
    """
    Reward-shaped terminal outcome.
    Outright win = ±1.0 (always dominates).
    Tiebreak win = +0.4, tiebreak loss = -0.4.
    Small progress bonus so the model prefers advanced positions at tiebreak,
    but can never make a tiebreak look better than an outright win.
    """
    if winner == 0:
        base = 0.0
    elif outright:
        base = 1.0 if winner == player else -1.0
    else:
        base = 0.4 if winner == player else -0.4

    def progress_bonus(pl):
        goal  = P1_GOAL if pl == 1 else P2_GOAL
        bonus = 0.0
        for r in range(ROWS):
            for c in range(COLS):
                p = board[r][c]
                if p and p.player == pl:
                    if p.has_crossed:      bonus += 0.01
                    if abs(r - goal) == 0: bonus += 0.02
        return bonus

    bonus_me  = progress_bonus(player)
    bonus_opp = progress_bonus(3 - player)
    shaped    = base + (bonus_me - bonus_opp) * 0.15
    return max(-1.0, min(1.0, shaped))


def player_progress_score(board, player):
    """
    Per-step positional signal active from move 1.
    Quadratic forward-progress for all pieces; bonus quadratic for crossed pieces.
    Creates value variation within a game so MCTS can distinguish good from bad positions
    even before any piece has crossed the midline.
    """
    goal    = P1_GOAL if player == 1 else P2_GOAL
    midline = ROWS // 2
    score   = 0.0
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r][c]
            if p and p.player == player:
                dist_to_goal = abs(r - goal)
                forward      = (ROWS - 1 - dist_to_goal) / (ROWS - 1)
                score       += forward ** 2
                if p.has_crossed:
                    score += 0.5 * (1.0 - dist_to_goal / midline) ** 2
    return score


def play_game(net1, net2, max_moves=MAX_MOVES, swap_start=False, warmup=False, temperature=1.0):
    """
    Play one game and return (state, pi, value) training samples.

    warmup=True  → greedy play (no network needed), one-hot pi.
    warmup=False → MCTS play using net1/net2, visit-count pi.

    Value label blends terminal shaped outcome (0.7) with per-step
    positional cubic score (0.3) for direct forward-progress signal.
    """
    gs         = GameState()
    if swap_start:
        gs.current_player = 2
    history    = []     # (tensor, pi, player, board_snapshot)
    nets       = {1: net1, 2: net2}
    pos_counts = {}

    for _ in range(max_moves):
        if gs.is_terminal(): break

        # Repetition detection (3-fold) — works correctly with FORWARD_BIAS
        key = (_board_key(gs.board), gs.current_player)
        pos_counts[key] = pos_counts.get(key, 0) + 1
        if pos_counts[key] >= 3: break

        player = gs.current_player
        tensor = gs.to_tensor()
        board  = [row[:] for row in gs.board]  # snapshot for step reward

        if warmup or nets[player] is None:
            move = greedy_move(gs)
            pi   = np.zeros(220, dtype=np.float32)
            if move:
                tr, tc  = move[2]['target']
                canon_r = (ROWS - 1 - tr) if player == 2 else tr
                pi[canon_r * 11 + tc] = 1.0
        else:
            result = run_mcts(gs, nets[player], n_sims=MCTS_SIMS,
                              add_noise=True, temperature=temperature)
            if result is None or result[0] is None: break
            move, pi = result

        if move is None: break
        history.append((tensor, pi, player, board))
        gs = gs.apply(*move)

    winner = gs.winner or tiebreak_winner(gs.board)

    # Value: undiscounted shaped outcome + small positional blend.
    # All positions in a won game get the same sign (~+0.65); step provides
    # small within-game variance (±0.15). MCTS needs a reliable win predictor,
    # not discounted future reward — discounting collapses labels toward zero.
    outright = bool(gs.winner)
    n_moves  = len(history)
    # Tempo penalty: reward decays slightly the longer the game goes.
    # At 200 moves the penalty is 0.25, at 100 moves 0.125, at 50 moves 0.0625.
    # Encourages decisive play — a fast win beats a slow win, a fast tiebreak
    # beats a slow one, making endless oscillation actively bad.
    tempo_penalty = min(0.25, n_moves / 800.0)
    samples = []
    for t, pi, p, b in history:
        final = np.tanh(shaped_outcome(winner, p, b, outright=outright)) - tempo_penalty
        step  = np.tanh(player_progress_score(b, p) - player_progress_score(b, 3 - p))
        val   = max(-1.0, min(1.0, 0.85 * final + 0.15 * step))
        samples.append((t, pi, val))

    return samples, bool(gs.winner)


# ── Elo helpers ───────────────────────────────────────────────
def get_expected_score(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))

def update_elo(ra, rb, score, k=ELO_K):
    return ra + k * (score - get_expected_score(ra, rb))


# ── Evaluate current vs best ──────────────────────────────────
def evaluate(current_net, best_net, n_games=EVAL_GAMES):
    current_wins  = 0
    outright_wins = 0
    draws         = 0
    for i in range(n_games):
        current_is_p1 = (i % 2 == 0)
        nets = ({1: current_net, 2: best_net} if current_is_p1
                else {1: best_net, 2: current_net})
        gs        = GameState()
        prog_win  = []
        pos_counts = {}
        for _ in range(MAX_MOVES):
            if gs.is_terminal(): break
            key = (_board_key(gs.board), gs.current_player)
            pos_counts[key] = pos_counts.get(key, 0) + 1
            if pos_counts[key] >= 3: break
            prog = total_progress(gs.board)
            prog_win.append(prog)
            if len(prog_win) > 40: prog_win.pop(0)
            if len(prog_win) == 40 and max(prog_win) <= prog_win[0]: break
            result = run_mcts(gs, nets[gs.current_player],
                              n_sims=MCTS_EVAL_SIMS, add_noise=False, temperature=0.5)
            if result is None or result[0] is None: break
            move, _ = result
            gs = gs.apply(*move)
        current_id = 1 if current_is_p1 else 2
        if gs.winner:
            if gs.winner == current_id:
                current_wins += 1; outright_wins += 1
        else:
            tb = tiebreak_winner(gs.board)
            if tb == current_id: current_wins += 1
            elif tb == 0:        draws += 1
    return current_wins / n_games, outright_wins, draws


# ── Evaluate current net vs minimax ───────────────────────────
def evaluate_vs_minimax(current_net, n_games=EVAL_GAMES):
    net_wins      = 0
    outright_wins = 0
    draws         = 0
    for i in range(n_games):
        net_is_p1 = (i % 2 == 0)
        gs        = GameState()
        prog_win  = []
        pos_counts = {}
        for _ in range(MAX_MOVES):
            if gs.is_terminal(): break
            key = (_board_key(gs.board), gs.current_player)
            pos_counts[key] = pos_counts.get(key, 0) + 1
            if pos_counts[key] >= 3: break
            prog = total_progress(gs.board)
            prog_win.append(prog)
            if len(prog_win) > 40: prog_win.pop(0)
            if len(prog_win) == 40 and max(prog_win) <= prog_win[0]: break
            if gs.current_player == (1 if net_is_p1 else 2):
                result = run_mcts(gs, current_net,
                                  n_sims=MCTS_EVAL_SIMS, add_noise=False, temperature=0.5)
                if result is None or result[0] is None: break
                move, _ = result
            else:
                move = minimax_move(gs, depth=2)
            if move is None: break
            gs = gs.apply(*move)
        net_id = 1 if net_is_p1 else 2
        if gs.winner:
            if gs.winner == net_id:
                net_wins += 1; outright_wins += 1
        else:
            tb = tiebreak_winner(gs.board)
            if tb == net_id: net_wins += 1
            elif tb == 0:    draws += 1
    return net_wins / n_games, outright_wins, draws


# ── Record one eval game for the UI ──────────────────────────
EVAL_GAMES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_games.json')

def board_snapshot(board):
    cells = []
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r][c]
            if p:
                cells.append({'r': r, 'c': c, 'pl': p.player,
                               'v': p.is_veylant, 'x': p.has_crossed})
    return cells

def record_eval_game(current_net, best_net, iteration, elo_current, elo_best, net_player=1):
    gs         = GameState()
    moves_log  = []
    boards_log = [board_snapshot(gs.board)]
    move_count = 0
    prog_win   = []
    nets       = {net_player: current_net, 3 - net_player: best_net}

    for _ in range(MAX_MOVES):
        if gs.is_terminal(): break
        prog = total_progress(gs.board)
        prog_win.append(prog)
        if len(prog_win) > 40: prog_win.pop(0)
        if len(prog_win) == 40 and max(prog_win) <= prog_win[0]: break
        result = run_mcts(gs, nets[gs.current_player],
                          n_sims=MCTS_EVAL_SIMS, add_noise=False, temperature=0.5)
        if result is None or result[0] is None: break
        (fr, fc, m), _ = result
        tr, tc = m['target']
        moves_log.append({'n': move_count + 1, 'pl': gs.current_player,
                          'fr': fr, 'fc': fc, 'tr': tr, 'tc': tc})
        gs = gs.apply(fr, fc, m)
        move_count += 1
        boards_log.append(board_snapshot(gs.board))

    winner   = gs.winner or tiebreak_winner(gs.board)
    end_type = 'outright' if gs.winner else ('tiebreak' if winner != 0 else 'tie')

    def stats(player):
        goal    = P1_GOAL if player == 1 else P2_GOAL
        crossed = sum(1 for r in range(ROWS) for c in range(COLS)
                      if gs.board[r][c] and gs.board[r][c].player == player
                      and gs.board[r][c].has_crossed)
        backline = sum(1 for c in range(COLS)
                       if gs.board[goal][c] and gs.board[goal][c].player == player)
        prog    = sum((ROWS - 1) - abs(r - goal)
                      + (ROWS if gs.board[r][c] and gs.board[r][c].has_crossed else 0)
                      for r in range(ROWS) for c in range(COLS)
                      if gs.board[r][c] and gs.board[r][c].player == player)
        return {'crossed': crossed, 'backline': backline, 'prog': prog}

    record = {'iter': iteration, 'winner': winner, 'end': end_type,
              'moves': move_count, 'p1': stats(1), 'p2': stats(2),
              'history': moves_log, 'boards': boards_log,
              'elo_current': round(elo_current, 1), 'elo_best': round(elo_best, 1),
              'net_player': net_player}

    games = []
    if os.path.exists(EVAL_GAMES_PATH):
        try:
            with open(EVAL_GAMES_PATH) as f: games = json.load(f)
        except Exception: games = []
    games.append(record)
    if len(games) > 5: games = games[-5:]
    with open(EVAL_GAMES_PATH, 'w') as f: json.dump(games, f)


# ── Main training loop ────────────────────────────────────────
def train():
    print(f'Device: {DEVICE} | Channels: {CHANNELS} | ResBlocks: {RES_BLOCKS}')
    print(f'MCTS sims: {MCTS_SIMS} (eval: {MCTS_EVAL_SIMS}) | batch: {MCTS_BATCH}')
    print(f'Checkpoint dir: {CHECKPOINT_DIR}\n')

    net  = StokcadeNet(CHANNELS, RES_BLOCKS).to(DEVICE)
    best = StokcadeNet(CHANNELS, RES_BLOCKS).to(DEVICE)

    best_path = os.path.join(CHECKPOINT_DIR, 'best.pt')
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=DEVICE)
        net.load_state_dict(state)
        best.load_state_dict(state)
        print(f'Resumed from {best_path}')
    else:
        best.load_state_dict(net.state_dict())
    skip_warmup = False  # always warm-up to recalibrate value head on pure win/loss labels

    params = sum(p.numel() for p in net.parameters())
    print(f'Model parameters: {params:,}\n')

    opt     = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    sched   = optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5)
    replay  = deque(maxlen=REPLAY_SIZE)

    start_iter    = 1
    in_warmup     = not skip_warmup
    rl_start_iter = None if in_warmup else 0
    total_g       = 0
    best_elo      = ELO_BASE
    curr_elo      = ELO_BASE
    avg_v_loss    = 999.0
    new_best_count = 0

    if in_warmup:
        print('Phase 1 — Greedy warm-up (filling replay buffer)...\n')
    else:
        print('Skipping warm-up (checkpoint loaded) — MCTS from iter 1.\n')

    # Temperature scheduler (decay exploration over time)
    def get_temp(it):
        if it < 20: return 1.0   # Full exploration during warm-up
        if it < 100: return 1.0 - (it - 20) * (0.8 / 80) # Linear decay to 0.2
        return 0.1 # Deterministic for late training phase

    try:
        for it in range(start_iter, 1000):
            with open(HEARTBEAT_FILE, 'w') as f:
                f.write(str(time.time()))

            # ── Warm-up exit check ─────────────────────────────
            if in_warmup:
                if it <= WARMUP_ITERS:
                    phase = 'greedy-warmup'
                else:
                    in_warmup     = False
                    rl_start_iter = it
                    replay.clear()
                    print(f'\nPhase 2 — MCTS RL begins at iter {it}. Buffer cleared.\n')
                    phase = 'mcts'
            else:
                phase = 'mcts'

            temp  = get_temp(it)
            print(f"\n[iter {it:3d}] [{phase:13s}] temp={temp:.2f} ...", flush=True)

            t0 = time.time()

            # ── Self-play ──────────────────────────────────────
            net.eval(); best.eval()
            outright_g = 0

            for g in range(GAMES_PER_ITER):
                n1, n2  = (net, best) if g % 2 == 0 else (best, net)
                samples, is_outright = play_game(
                    None if in_warmup else n1,
                    None if in_warmup else n2,
                    swap_start=(g % 2 == 1),
                    warmup=in_warmup,
                    temperature=temp,
                )
                if is_outright:
                    outright_g += 1
                replay.extend(samples)
                total_g += 1

            # Inject 1 greedy anchor game every MCTS iter to keep decisive
            # win/loss signal in the buffer and prevent value head collapse.
            if not in_warmup:
                anchor_samples, _ = play_game(None, None, warmup=True)
                replay.extend(anchor_samples)

            # ── Train ──────────────────────────────────────────
            if len(replay) < BATCH_SIZE:
                elapsed = time.time() - t0
                print(f'[iter {it:3d}] Collecting ({len(replay):>5}/{BATCH_SIZE}) ... {elapsed:.1f}s', flush=True)
                continue

            net.train()
            total_loss = 0.0
            total_v    = 0.0
            total_p    = 0.0

            for _ in range(TRAIN_STEPS):
                batch    = random.sample(replay, BATCH_SIZE)
                states   = torch.tensor(np.array([s for s, _, _ in batch]),
                                        dtype=torch.float32).to(DEVICE)
                target_p = torch.tensor(np.array([pi for _, pi, _ in batch]),
                                        dtype=torch.float32).to(DEVICE)
                target_v = torch.tensor([v for _, _, v in batch],
                                        dtype=torch.float32).to(DEVICE)

                opt.zero_grad()
                p_out, v_out = net(states)

                v_loss = nn.MSELoss()(v_out.squeeze(-1), target_v)

                # Cross-entropy: -sum(pi * log_softmax(logits))
                # Clamp to avoid NaN from log(~0)
                log_pi = torch.clamp(torch.log_softmax(p_out, dim=1), min=-20.0)
                p_loss = -torch.mean(torch.sum(target_p * log_pi, dim=1))

                loss = v_loss * 3.0 + p_loss
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 2.0)
                opt.step()

                total_loss += loss.item()
                total_v    += v_loss.item()
                total_p    += p_loss.item()

            sched.step()

            elapsed    = time.time() - t0
            avg_loss   = total_loss / TRAIN_STEPS
            avg_v_loss = total_v    / TRAIN_STEPS
            avg_p_loss = total_p    / TRAIN_STEPS

            print(f'  games={total_g:5d} | replay={len(replay):6d} | '
                  f'V={avg_v_loss:.4f} P={avg_p_loss:.4f} | '
                  f'Elo={curr_elo:.1f} | wins={outright_g}/{GAMES_PER_ITER} | {elapsed:.1f}s')
            sys.stdout.flush()

            torch.save(net.state_dict(), os.path.join(CHECKPOINT_DIR, 'latest.pt'))

            # ── Evaluate ───────────────────────────────────────
            if it % EVAL_EVERY == 0 and not in_warmup:
                print(f'  → Evaluating ({EVAL_GAMES} games)...', flush=True)
                wr,    out_b,  draws_b  = evaluate(net, best, n_games=EVAL_GAMES)
                wr_mm, out_mm, draws_mm = evaluate_vs_minimax(net, n_games=EVAL_GAMES)
                curr_elo = update_elo(curr_elo, best_elo, wr)
                print(f'  vs best: {wr:.1%} (out:{out_b}) | vs minimax: {wr_mm:.1%} (out:{out_mm}) | Elo:{curr_elo:.1f}', end='')

                net_side = 1 if (it // EVAL_EVERY) % 2 == 0 else 2
                record_eval_game(net, best, it, curr_elo, best_elo, net_player=net_side)

                if wr >= WIN_THRESHOLD:
                    best.load_state_dict(net.state_dict())
                    torch.save(net.state_dict(), best_path)
                    best_elo = curr_elo
                    new_best_count += 1
                    if new_best_count % 3 == 0:
                        snap = os.path.join(CHECKPOINT_DIR, f'snapshot_iter{it:04d}.pt')
                        shutil.copy2(best_path, snap)
                    print(f'  ✓ New best! Elo: {best_elo:.1f}')
                else:
                    print(f'  (need {WIN_THRESHOLD:.0%})')

            elif it % EVAL_EVERY == 0:
                print(f'  (warm-up — eval starts once MCTS begins)')

    except KeyboardInterrupt:
        print('\nStopped. Saving latest...')
        torch.save(net.state_dict(), os.path.join(CHECKPOINT_DIR, 'latest.pt'))
        print(f'Saved to {CHECKPOINT_DIR}/latest.pt')


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    train()
