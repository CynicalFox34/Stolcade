import math
import torch
import numpy as np
from copy import deepcopy

class MCTSNode:
    def __init__(self, state, p=0, parent=None):
        self.state = state
        self.parent = parent
        self.children = {}  # move_tuple -> MCTSNode
        self.n = 0          # visit count
        self.w = 0          # total value
        self.q = 0          # mean value (w/n)
        self.p = p          # prior probability

    def is_expanded(self):
        return len(self.children) > 0

class MCTS:
    def __init__(self, model, device='cpu', cpuct=1.5):
        self.model = model
        self.device = device
        self.cpuct = cpuct

    @torch.no_grad()
    def search(self, state, num_simulations=100):
        # Create root node
        root = MCTSNode(state)

        for _ in range(num_simulations):
            node = root
            search_path = [node]

            # 1. Select
            while node.is_expanded():
                move, node = self._select_child(node)
                search_path.append(node)

            # 2. Expand & Evaluate
            value = self._expand_and_evaluate(node)

            # 3. Backpropagate
            self._backpropagate(search_path, value, node.state.current_player)

        # Return move probabilities based on visit counts
        return self._get_pi(root)

    def _select_child(self, node):
        best_score = -float('inf')
        best_move = None
        best_child = None

        total_n = sum(child.n for child in node.children.values())
        
        for move, child in node.children.items():
            # AlphaZero PUCT formula: Q + c_puct * P * (sqrt(N) / (1 + n))
            u_score = child.q + self.cpuct * child.p * (math.sqrt(total_n) / (1 + child.n))
            if u_score > best_score:
                best_score = u_score
                best_move = move
                best_child = child
        
        return best_move, best_child

    def _expand_and_evaluate(self, node):
        # Check for terminal state
        if node.state.is_terminal():
            winner = node.state.winner
            if winner == 0: return 0
            return 1 if winner == node.state.current_player else -1

        # Use model to get Policy (p) and Value (v)
        tensor = torch.tensor(node.state.to_tensor(), dtype=torch.float32).unsqueeze(0).to(self.device)
        p_logits, v = self.model(tensor)
        
        # Policy is for target squares (20x11 = 220)
        p_probs = torch.softmax(p_logits, dim=1).cpu().numpy()[0]
        value = v.cpu().item()

        # Expand node with legal moves
        legal_moves = node.state.legal_moves()
        if not legal_moves: return 0

        # Map Policy to legal moves
        # For now, we take the average probability of the target square for that move
        move_probs = []
        for move in legal_moves:
            # move format: (r, c, move_obj)
            # move_obj has 'target': (tr, tc)
            tr, tc = move[2]['target']
            move_probs.append(p_probs[tr * 11 + tc])
        
        # Renormalize move probabilities
        sum_p = sum(move_probs) + 1e-8
        move_probs = [p / sum_p for p in move_probs]

        for i, move in enumerate(legal_moves):
            # move tuple for hashability: (from_r, from_c, dr, dc, is_veylant)
            m_obj = move[2]
            m_tuple = (move[0], move[1], m_obj.get('dr', 0), m_obj.get('dc', 0), node.state.board[move[0]][move[1]].is_veylant)
            
            # Apply move to get next state
            next_state = node.state.apply(*move)
            node.children[m_tuple] = MCTSNode(next_state, p=move_probs[i], parent=node)

        return value

    def _backpropagate(self, search_path, value, root_player):
        for node in reversed(search_path):
            # Value is relative to node.state.current_player
            # If the value comes from a state where it's the current player's turn,
            # we need to flip it if the backprop player is different.
            v = value if node.state.current_player == root_player else -value
            node.n += 1
            node.w += v
            node.q = node.w / node.n

    def _get_pi(self, root):
        # pi is proportional to visit counts: pi(a|s) = N(s,a)^(1/tau)
        # We use tau=1 for now
        moves = list(root.children.keys())
        counts = [child.n for child in root.children.values()]
        total = sum(counts)
        probs = [c / total for c in counts]
        return moves, probs
