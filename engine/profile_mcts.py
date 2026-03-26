"""Quick profiling script to find the MCTS bottleneck."""
import os, sys, time, cProfile, pstats, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from game import GameState, tiebreak_winner, P1_GOAL, P2_GOAL, ROWS, COLS
from model import StokcadeNet
import train  # imports everything from train

DEVICE = train.DEVICE
CHECKPOINT = os.path.join(train.CHECKPOINT_DIR, 'best.pt')

print(f"Device: {DEVICE}")
print(f"MCTS_SIMS={train.MCTS_SIMS}  MCTS_BATCH={train.MCTS_BATCH}  FORWARD_BIAS={train.FORWARD_BIAS}")

# Load model
net = StokcadeNet().to(DEVICE)
net.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
net.eval()
print(f"Loaded {CHECKPOINT}\n")

# ── Time a single MCTS call ──────────────────────────────────
gs = GameState()
t0 = time.time()
move, pi = train.run_mcts(gs, net, n_sims=train.MCTS_SIMS, add_noise=True, temperature=1.0)
t1 = time.time()
print(f"Single run_mcts() call: {t1-t0:.2f}s  (move={move})")

# ── Time apply() and get_hash() ────────────────────────────────
N = 1000
gs2 = GameState()
moves = gs2.legal_moves()
t0 = time.time()
for _ in range(N):
    gs2.apply(*moves[0])
t1 = time.time()
print(f"apply() × {N}: {(t1-t0)*1000/N:.3f}ms each")

t0 = time.time()
for _ in range(N):
    gs2.get_hash()
t1 = time.time()
print(f"get_hash() × {N}: {(t1-t0)*1000/N:.3f}ms each")

t0 = time.time()
for _ in range(N):
    gs2.legal_moves()
t1 = time.time()
print(f"legal_moves() × {N}: {(t1-t0)*1000/N:.3f}ms each")

t0 = time.time()
for _ in range(N):
    gs2.to_tensor()
t1 = time.time()
print(f"to_tensor() × {N}: {(t1-t0)*1000/N:.3f}ms each")

# ── Play one full game and measure length + time ────────────────
print("\nPlaying 3 full MCTS games to measure game length...")
for g in range(3):
    gs = GameState()
    moves_played = 0
    t0 = time.time()
    while not gs.is_terminal() and moves_played < train.MAX_MOVES:
        key = (id(gs.board), gs.current_player)  # dummy rep check
        move, pi = train.run_mcts(gs, net, n_sims=train.MCTS_SIMS, add_noise=False, temperature=0.0)
        if move is None:
            break
        gs = gs.apply(*move)
        moves_played += 1
    elapsed = time.time() - t0
    winner = gs.winner or tiebreak_winner(gs.board)
    print(f"  Game {g+1}: {moves_played} moves, {elapsed:.1f}s, {elapsed/max(moves_played,1):.2f}s/move, winner={winner}")

# ── cProfile on a single game ───────────────────────────────────
print("\nRunning cProfile on 1 game (50 sims for speed)...")
pr = cProfile.Profile()
gs = GameState()
pr.enable()
for _ in range(30):  # 30 moves
    if gs.is_terminal(): break
    move, pi = train.run_mcts(gs, net, n_sims=50, add_noise=False, temperature=0.0)
    if move is None: break
    gs = gs.apply(*move)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(25)
print(s.getvalue())
