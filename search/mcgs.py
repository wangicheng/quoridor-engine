import math
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import core and nn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.env import QuoridorEnv
from nn.model import QuoridorNet

class MCGSNode:
    def __init__(self, prior_prob):
        self.prior_prob = prior_prob
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_value = 0.0 # Stored from NN evaluation for FPU
        self.children = {} # action_id -> MCGSNode
        
    def value(self):
        if self.visit_count == 0:
            return self.prior_value
        return self.value_sum / self.visit_count

class MCGS:
    def __init__(self, model_p1, model_p2=None, c_puct=1.5, fpu_reduction=0.01):
        self.model_p1 = model_p1
        self.model_p2 = model_p2 if model_p2 is not None else model_p1
        self.c_puct = c_puct
        self.fpu_reduction = fpu_reduction
        self.transposition_table = {} # state_hash -> MCGSNode
        self.device = next(model_p1.parameters()).device
        
    def _evaluate(self, env):
        model = self.model_p1 if env.turn == 0 else self.model_p2
        state, mask = env.get_state()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mask_tensor = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, nn_value, _, _ = model(state_tensor, mask_tensor)
            probs = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
            nn_value = nn_value.item()
            
        p1_dist = env.last_p1_dist
        p2_dist = env.last_p2_dist
        
        if env.turn == 0:
            heuristic_value = math.tanh((p2_dist - p1_dist) / 3.0)
        else:
            heuristic_value = math.tanh((p1_dist - p2_dist) / 3.0)
            
        # Mix NN Value with Heuristic Value
        # 0.4 weight ensures the heuristic guides the agent without forcing it to greedily dump all walls
        final_value = 0.8 * nn_value + 0.4 * heuristic_value
            
        return probs, final_value, mask

    def search(self, initial_env, num_simulations=100):
        root_hash = initial_env.get_hash()
        
        if root_hash not in self.transposition_table:
            self.transposition_table[root_hash] = MCGSNode(prior_prob=1.0)
            
        root = self.transposition_table[root_hash]
        
        if len(root.children) == 0:
            probs, value, mask = self._evaluate(initial_env)
            root.prior_value = value
            valid_moves = int(mask.sum())
            if valid_moves > 0:
                noise = np.random.dirichlet([0.3] * valid_moves)
                noise_idx = 0
                for action in range(132):
                    if mask[action]:
                        p = 0.75 * probs[action] + 0.25 * noise[noise_idx]
                        root.children[action] = MCGSNode(prior_prob=p)
                        noise_idx += 1
            root.visit_count += 1
            root.value_sum += value
            
        for _ in range(num_simulations):
            env = initial_env.clone()
            self._simulate(root, env, visited=set())
            
        action_visits = np.zeros(132)
        for action, child in root.children.items():
            action_visits[action] = child.visit_count
            
        return action_visits

    def _simulate(self, node, env, visited):
        current_hash = env.get_hash()
        if current_hash in visited:
            return -1.0
            
        visited.add(current_hash)
        
        if env.is_terminal():
            val = env.get_result(env.turn)
            node.visit_count += 1
            node.value_sum += val
            return -val

        if len(node.children) == 0:
            probs, value, mask = self._evaluate(env)
            node.prior_value = value
            
            for action in range(132):
                if mask[action]:
                    node.children[action] = MCGSNode(prior_prob=probs[action])
                    
            node.visit_count += 1
            node.value_sum += value
            return -value
            
        best_action = -1
        best_ucb = -float('inf')
        
        sqrt_total_visits = math.sqrt(node.visit_count)
        
        for action, child in node.children.items():
            if child.visit_count == 0:
                # AlphaZero FPU: Use the parent's neural network value (prior_value)
                # Since prior_value is from parent's perspective, we don't negate it.
                q_val = node.prior_value
                # Penalize walls slightly to encourage pawn moves early on
                if action >= 4:
                    q_val -= self.fpu_reduction
            else:
                # child.value() is from the child's perspective. Must negate it!
                q_val = -child.value()
                
            u_val = self.c_puct * child.prior_prob * sqrt_total_visits / (1 + child.visit_count)
            ucb = q_val + u_val
            
            if ucb > best_ucb:
                best_ucb = ucb
                best_action = action
                
        env.step(best_action)
        next_hash = env.get_hash()
        
        if next_hash in self.transposition_table:
            node.children[best_action] = self.transposition_table[next_hash]
        else:
            self.transposition_table[next_hash] = node.children[best_action]
            
        next_node = node.children[best_action]
        
        val = self._simulate(next_node, env, visited)
        
        node.visit_count += 1
        node.value_sum += val
        
        return -val

if __name__ == "__main__":
    print("Testing Python MCGS...")
    device = torch.device("cpu")
    model = QuoridorNet().to(device)
    model.eval() # Set to evaluation mode
    
    env = QuoridorEnv()
    mcgs = MCGS(model)
    
    print("Running 50 simulations from initial state...")
    action_visits = mcgs.search(env, num_simulations=50)
    
    best_action = np.argmax(action_visits)
    print(f"Visits Distribution: {action_visits[action_visits > 0]}")
    print(f"Best Action ID: {best_action}")
    if best_action < 4:
        print("Action Type: Pawn Move")
    else:
        print("Action Type: Wall Placement")
        
    print(f"Transposition Table Size (Unique DAG Nodes): {len(mcgs.transposition_table)}")
