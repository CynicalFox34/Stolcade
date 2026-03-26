"""
Quick sanity check — plays 20 moves with the current network and reports:
- Are pieces moving forward (toward goal)?
- Are pieces crossing the midline?
- What does the value head output (should vary between positions)?
- Are move selections decisive or uniform (random)?

Run: cd engine && python3 sanity_check.py
"""
import os, sys, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game import GameState, P1_GOAL, P2_GOAL, ROWS
from model import StokcadeNet
from train import CHANNELS, RES_BLOCKS, NET_DEPTH_TRAIN, DEVICE

CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'best.pt')

def net_move_simple(gs, net):
    moves = gs.legal_moves()
    if not moves: return None
    states = [gs.apply(fr, fc, m) for fr, fc, m in moves]
    batch  = torch.tensor(np.array([s.to_tensor() for s in states]),
                          dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        _, v_out = net(batch)
    vals = v_out.squeeze(-1).cpu().numpy()
    # Flip for opponent states
    for i, s in enumerate(states):
        if s.current_player != gs.current_player:
            vals[i] = -vals[i]
    best_idx = int(np.argmax(vals))
    return moves[best_idx], vals

net = StokcadeNet(CHANNELS, RES_BLOCKS).to(DEVICE)
if os.path.exists(CHECKPOINT):
    net.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    print(f"Loaded {CHECKPOINT}")
else:
    print("No checkpoint found — using random weights")
net.eval()

gs = GameState()
print(f"\nPlaying 30 moves. Tracking forward progress.\n")
print(f"{'Move':>4} {'Player':>6} {'From':>8} {'To':>8} {'Direction':>12} {'Val range':>12}")
print("-" * 60)

forward_moves = 0
backward_moves = 0
sideways_moves = 0
crossings = 0
val_outputs = []

for i in range(30):
    result = net_move_simple(gs, net)
    if result is None: break
    (fr, fc, m), vals = result
    tr, tc = m['target']
    val_outputs.extend(vals.tolist())

    player = gs.current_player
    goal   = P1_GOAL if player == 1 else P2_GOAL
    before_dist = abs(fr - goal)
    after_dist  = abs(tr - goal)

    if after_dist < before_dist:
        direction = "FORWARD ✓"
        forward_moves += 1
    elif after_dist > before_dist:
        direction = "BACKWARD ✗"
        backward_moves += 1
    else:
        direction = "sideways"
        sideways_moves += 1

    crossed = (player == 1 and fr < 10 and tr >= 10) or (player == 2 and fr >= 10 and tr < 10)
    if crossed:
        direction += " CROSS!"
        crossings += 1

    print(f"{i+1:>4} {'P'+str(player):>6} ({fr},{fc})→({tr},{tc})  {direction:>20}  val∈[{vals.min():.2f},{vals.max():.2f}]")
    gs = gs.apply(fr, fc, m)

print("\n── Summary ──────────────────────────────────────")
print(f"Forward moves:  {forward_moves}/30")
print(f"Sideways moves: {sideways_moves}/30")
print(f"Backward moves: {backward_moves}/30")
print(f"Crossings:      {crossings}")
print(f"Value output range: [{min(val_outputs):.3f}, {max(val_outputs):.3f}]")
print(f"Value output std:   {np.std(val_outputs):.3f}  (near 0 = stuck, >0.1 = learning)")
print()
if backward_moves > 10:
    print("⚠ TOO MANY BACKWARD MOVES — network not learning forward progress")
elif forward_moves < 10:
    print("⚠ NOT ENOUGH FORWARD MOVES — check reward signal")
else:
    print("✓ Network is making forward progress")

if np.std(val_outputs) < 0.05:
    print("⚠ VALUE OUTPUT IS NEAR CONSTANT — value head not working")
else:
    print("✓ Value head producing varied outputs")
