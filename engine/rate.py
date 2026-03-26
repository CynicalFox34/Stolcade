"""
Stolcade Round-Robin Rating Script
===================================
Runs a round-robin tournament between all snapshot checkpoints plus
minimax bots at configurable depths, then computes Elo ratings.

Usage:
    python rate.py                    # all snapshots + minimax depths 1,2,3
    python rate.py --games 20         # 20 games per pair (default: 10)
    python rate.py --minimax 1 2 3 4  # minimax depths to include
    python rate.py --no-minimax       # snapshots only
    python rate.py --sims 100         # MCTS sims per move (default: 50)
"""

import os, sys, argparse, glob, itertools, random
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from game import GameState, did_cross, P1_GOAL, P2_GOAL, ROWS, COLS, tiebreak_winner
from model import StokcadeNet
from mcts import MCTSNode, mcts_move
from minimax import minimax_move

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
MAX_MOVES      = 300
DEVICE         = 'mps' if torch.backends.mps.is_available() else \
                 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Minimax anchor Elos (tune these as you calibrate) ──────────────────────
MINIMAX_ANCHOR = {1: 600, 2: 800, 3: 1000}


def load_net(path):
    net = StokcadeNet().to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net


def play_game(p1_agent, p2_agent, max_moves=MAX_MOVES):
    """Play one game. Agents are callables: agent(gs) -> move."""
    gs = GameState()
    for _ in range(max_moves):
        agent = p1_agent if gs.current_player == 1 else p2_agent
        move  = agent(gs)
        if move is None:
            break
        gs.apply_move(*move)
        if gs.winner:
            return gs.winner
    return tiebreak_winner(gs)


def net_agent(net, sims):
    def agent(gs):
        return mcts_move(gs, net, num_simulations=sims, temperature=0.0)
    return agent


def mm_agent(depth):
    def agent(gs):
        return minimax_move(gs, depth=depth)
    return agent


def expected(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))


def compute_elo(results, init_elos, k=32, iterations=200):
    """
    Iterative Elo computation.
    results: list of (name_a, name_b, score_a)  where score_a in {0, 0.5, 1}
    init_elos: dict name -> starting Elo (anchors are held fixed)
    """
    elos   = dict(init_elos)
    anchor = set(k for k, v in init_elos.items() if isinstance(v, int) and 'minimax' in k)

    for _ in range(iterations):
        random.shuffle(results)
        for a, b, score_a in results:
            ea = expected(elos[a], elos[b])
            eb = 1.0 - ea
            if a not in anchor:
                elos[a] += k * (score_a       - ea)
            if b not in anchor:
                elos[b] += k * ((1 - score_a) - eb)
    return elos


def run(args):
    print(f'Device: {DEVICE}')

    # ── Collect participants ───────────────────────────────────────────────
    snap_paths = sorted(glob.glob(os.path.join(CHECKPOINT_DIR, 'snapshot_iter*.pt')))
    if not snap_paths:
        print('No snapshots found. Train a few iters first (snapshots save on each new best).')
        sys.exit(1)

    participants = {}  # name -> agent callable or 'net:<path>'
    init_elos    = {}

    for path in snap_paths:
        name = os.path.splitext(os.path.basename(path))[0]  # e.g. snapshot_iter0015
        participants[name] = ('net', path)
        init_elos[name]    = 1000

    if not args.no_minimax:
        for d in args.minimax:
            name = f'minimax_depth{d}'
            participants[name] = ('mm', d)
            init_elos[name]    = MINIMAX_ANCHOR.get(d, 800 + d * 150)

    names = list(participants.keys())
    print(f'\nParticipants ({len(names)}):')
    for n in names:
        print(f'  {n}  (start Elo: {init_elos[n]})')

    # ── Load nets (cache to avoid reloading) ──────────────────────────────
    net_cache = {}
    def get_agent(name):
        kind, val = participants[name]
        if kind == 'net':
            if val not in net_cache:
                print(f'  Loading {name}...')
                net_cache[val] = load_net(val)
            return net_agent(net_cache[val], args.sims)
        else:
            return mm_agent(val)

    # ── Round-robin ────────────────────────────────────────────────────────
    pairs   = list(itertools.combinations(names, 2))
    results = []
    total_games = len(pairs) * args.games
    played      = 0

    print(f'\nRunning {len(pairs)} matchups × {args.games} games = {total_games} total games...\n')

    for a, b in pairs:
        a_wins = b_wins = draws = 0
        agent_a = get_agent(a)
        agent_b = get_agent(b)

        for g in range(args.games):
            # Alternate who plays P1
            if g % 2 == 0:
                winner = play_game(agent_a, agent_b)
                if   winner == 1: a_wins += 1
                elif winner == 2: b_wins += 1
                else:             draws  += 1
            else:
                winner = play_game(agent_b, agent_a)
                if   winner == 2: a_wins += 1
                elif winner == 1: b_wins += 1
                else:             draws  += 1
            played += 1
            print(f'\r  [{played}/{total_games}] {a} vs {b}: {a_wins}W {b_wins}L {draws}D', end='', flush=True)

        print()
        score_a = (a_wins + 0.5 * draws) / args.games
        results.append((a, b, score_a))
        print(f'  → {a}: {score_a:.1%}  |  {b}: {1-score_a:.1%}')

    # ── Elo calculation ────────────────────────────────────────────────────
    print('\nComputing Elo ratings...')
    elos = compute_elo(results, init_elos)

    print('\n' + '='*50)
    print('  FINAL ELO RATINGS')
    print('='*50)
    for name, elo in sorted(elos.items(), key=lambda x: -x[1]):
        tag = ' [anchor]' if 'minimax' in name else ''
        print(f'  {name:<30} {elo:7.1f}{tag}')
    print('='*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--games',      type=int,   default=10,        help='Games per pair')
    parser.add_argument('--sims',       type=int,   default=50,        help='MCTS sims per move')
    parser.add_argument('--minimax',    type=int,   nargs='+', default=[1,2,3], help='Minimax depths')
    parser.add_argument('--no-minimax', action='store_true',           help='Skip minimax bots')
    args = parser.parse_args()
    run(args)
