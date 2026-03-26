"""
Stolcade supervised pretrainer.

Phase 0 — Pure supervised learning on minimax games.
  - Generate N minimax-vs-minimax games, alternating who starts
  - Label each position as win (+1) or loss (-1) for the current player
  - Train the network on this balanced dataset
  - Save as checkpoints/best.pt so train.py picks it up for RL

Run:
  cd engine
  python3 pretrain.py
Then:
  python3 train.py   (will load the pretrained weights and start RL)
"""

import os, sys, time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from game import GameState, tiebreak_winner, ROWS, COLS
from model import StokcadeNet
from minimax import minimax_move, _board_key

# ─────── Config ────────────────────────────────────────────────
DEVICE         = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
CHANNELS       = 128
RES_BLOCKS     = 12
NUM_GAMES      = 20
MAX_MOVES      = 400
STALL_LIMIT    = 40
BATCH_SIZE     = 128
EPOCHS         = 10
LR             = 1e-3
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
LOG_FILE       = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training.log')
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

def total_progress(board):
    total = 0
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r][c]
            if p:
                goal = 19 if p.player == 1 else 0
                total += (ROWS - 1) - abs(r - goal)
                if p.has_crossed: total += ROWS
    return total

def play_minimax_game(depth=3, swap_start=False):
    """
    Play one minimax-vs-minimax game.
    swap_start=True: P2 moves first, giving P2 the first-mover tiebreak advantage.
    Alternating swap_start across games produces a balanced ~50/50 dataset.
    """
    gs          = GameState()
    if swap_start:
        gs.current_player = 2
    history     = []
    pos_history = []
    stall       = 0
    last_prog   = total_progress(gs.board)

    for _ in range(MAX_MOVES):
        if gs.is_terminal(): break
        key = _board_key(gs.board)
        pos_history.append(key)
        if len(pos_history) > 12: pos_history.pop(0)

        move = minimax_move(gs, depth=depth, pos_history=pos_history)
        if move is None: break

        pi = np.zeros(220, dtype=np.float32)
        tr, tc = move[2]['target']
        pi[tr * 11 + tc] = 1.0

        history.append((gs.to_tensor(), gs.current_player, pi))
        gs = gs.apply(*move)

        prog = total_progress(gs.board)
        if prog > last_prog:
            last_prog = prog; stall = 0
        else:
            stall += 1
            if stall >= STALL_LIMIT: break

    winner = gs.winner or tiebreak_winner(gs.board)
    if not winner:
        return [], 0

    return [
        (tensor, pi, 1.0 if player == winner else -1.0)
        for tensor, player, pi in history
    ], winner

def main():
    print(f"=== Stolcade Supervised Pretrainer ===")
    print(f"Device: {DEVICE} | Channels: {CHANNELS} | ResBlocks: {RES_BLOCKS}")
    print()

    # ── Step 1: Generate minimax games (alternating start for balance) ──
    print(f"Generating {NUM_GAMES} minimax games (depth 2, alternating start)...")
    t0 = time.time()
    dataset  = []
    wins1, wins2 = 0, 0
    for i in range(NUM_GAMES):
        swap = (i % 2 == 1)   # odd games: P2 starts
        samples, winner = play_minimax_game(depth=2, swap_start=swap)
        if not samples:
            continue
        dataset.extend(samples)
        if winner == 1: wins1 += 1
        else:           wins2 += 1
        elapsed = time.time() - t0
        print(f"  game {i+1}/{NUM_GAMES} | positions: {len(dataset)} | {elapsed:.0f}s | W1:{wins1} W2:{wins2}")

    print(f"\nDataset: {len(dataset)} positions from {NUM_GAMES} games")
    print(f"P1 wins: {wins1} | P2 wins: {wins2}")
    print()

    # ── Step 2: Train ───────────────────────────────────────────
    net     = StokcadeNet(CHANNELS, RES_BLOCKS).to(DEVICE)
    opt     = optim.Adam(net.parameters(), lr=LR, weight_decay=1e-4)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    n         = len(dataset)
    best_loss = float('inf')

    print(f"Training for {EPOCHS} epochs over {n} positions...")
    for epoch in range(1, EPOCHS + 1):
        net.train()
        epoch_loss = 0.0
        steps = 0

        indices = np.random.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            idx   = indices[start:start+BATCH_SIZE]
            s_bat = torch.tensor(np.array([dataset[i][0] for i in idx]), dtype=torch.float32).to(DEVICE)
            p_bat = torch.tensor(np.array([dataset[i][1] for i in idx]), dtype=torch.float32).to(DEVICE)
            v_bat = torch.tensor(np.array([dataset[i][2] for i in idx]), dtype=torch.float32).to(DEVICE).view(-1, 1)

            opt.zero_grad()
            p_out, v_out = net(s_bat)
            v_loss = nn.MSELoss()(v_out, v_bat)
            p_loss = -torch.mean(torch.sum(p_bat * torch.log_softmax(p_out, dim=1), dim=1))
            loss   = p_loss + v_loss * 5.0
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()
            epoch_loss += loss.item()
            steps += 1

        sched.step()
        avg = epoch_loss / steps
        marker = ' ✓ best' if avg < best_loss else ''
        if avg < best_loss:
            best_loss = avg
            torch.save(net.state_dict(), os.path.join(CHECKPOINT_DIR, 'best.pt'))
        print(f"  Epoch {epoch:2d}/{EPOCHS} | loss={avg:.4f}{marker}")
        sys.stdout.flush()

    torch.save(net.state_dict(), os.path.join(CHECKPOINT_DIR, 'latest.pt'))
    print(f"\nDone. Saved to checkpoints/best.pt + checkpoints/latest.pt")
    print("Now run: python3 train.py")

if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
