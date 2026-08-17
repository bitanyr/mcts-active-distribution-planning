import math
import numpy as np
from core.node import MCTSNode

class MCTS:
    def __init__(self, neural_net, num_simulations=400):
        self.neural_net = neural_net
        self.num_simulations = num_simulations
        self.device_types = ['ess', 'gas', 'svc', 'cb']
        self.device_map = {'ess': 0, 'gas': 1, 'svc': 2, 'cb': 3}
        self.c_puct = 1.5

    def search(self, initial_state, temperature=1.0, add_noise=True):
        root = MCTSNode(state=initial_state.clone())
        self.evaluate_and_expand(root)
        
        # 1. Exploration
        if add_noise and root.children:
            actions = list(root.children.keys())
            noise = np.random.dirichlet([0.08] * len(actions))
            for i, action in enumerate(actions):
                root.children[action].prior_prob = 0.7 * root.children[action].prior_prob + 0.3 * noise[i]

        # 2. 
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            
            while node.is_expanded():
                action, node = self.select_child(node)
                search_path.append(node)
            
            value = self.evaluate_and_expand(node)
            
            self.backpropagate(search_path, value)

        # 3. Policy Extraction
        action_visits = {}
        for action, child in root.children.items():
            action_visits[action] = child.visit_count

        if not action_visits:
            return None, None
            
        actions = list(action_visits.keys())
        counts = list(action_visits.values())
        
        # 4. Temperature Scaling
        if temperature == 0:
            best_action = actions[np.argmax(counts)]
            probs = [1.0 if a == best_action else 0.0 for a in actions]
        else:
            counts = np.array(counts, dtype=np.float64) ** (1.0 / temperature)
            sum_counts = np.sum(counts)
            if sum_counts > 0:
                probs = counts / sum_counts
            else:
                probs = np.ones_like(counts) / len(counts)
            best_action = actions[np.random.choice(len(actions), p=probs)]

        
        pi_vector = np.zeros(self.neural_net.policy_dim)
        for i, action in enumerate(actions):
            if action[0] == 'stop':
                idx = self.neural_net.policy_dim - 1  
            else:
                d_type, bus = action
                idx = bus * 4 + self.device_map[d_type]
            pi_vector[idx] = probs[i]

        return best_action, pi_vector

    def select_child(self, node):
        best_score = -float('inf')
        best_action = None
        best_child = None

        FPU_PENALTY = 0.1 

        for action, child in node.children.items():
            if child.visit_count == 0:
                q_val = node.q_value - FPU_PENALTY
            else:
                q_val = child.q_value
                
            u_value = self.c_puct * child.prior_prob * math.sqrt(max(1, node.visit_count)) / (1 + child.visit_count)
            ucb_score = q_val + u_value
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        return best_action, best_child

    def evaluate_and_expand(self, node):
        if node.state is None:
            node.state = node.parent.state.clone()
            if node.action is not None and node.action[0] != 'stop':
                node.state.add_device(node.action[0], node.action[1])

        if node.action is not None and node.action[0] == 'stop':
             return node.parent.q_value if node.parent else 0.0

        
        state_tensor = self.state_to_tensor(node.state)
        action_probs, value, perf_index = self.neural_net.predict(state_tensor)
        node.perf_index = perf_index

        valid_actions = self.get_valid_actions(node.state)
        if not valid_actions:
            return value

        valid_probs = {}
        total_valid_p = 0.0
        for action in valid_actions:
            if action[0] == 'stop':
                idx = self.neural_net.policy_dim - 1
            else:
                d_type, bus = action
                idx = bus * 4 + self.device_map[d_type]
            p = action_probs[idx]
            valid_probs[action] = p
            total_valid_p += p
            
        # Action Masking Resolution
        for action in valid_actions:
            normalized_p = valid_probs[action] / total_valid_p if total_valid_p > 0 else 1.0/len(valid_actions)
            node.children[action] = MCTSNode(state=None, parent=node, action=action, prior_prob=normalized_p)
            
        return value


    def backpropagate(self, search_path, value):
        for node in search_path:
            node.visit_count += 1
            node.value_sum += value

    def get_valid_actions(self, state):
        valid_actions = []
        bus_equipment_count = {bus: 0 for bus in range(1, state.num_buses)}
        for d_type in self.device_types:
            for bus in state.placements[d_type]:
                bus_equipment_count[bus] += 1
                
        for d_type in self.device_types:
            for bus in range(1, state.num_buses):
                if bus not in state.placements[d_type] and bus_equipment_count[bus] < 2:
                    valid_actions.append((d_type, bus))
                    
        valid_actions.append(('stop', 0)) 
        return valid_actions

    def state_to_tensor(self, state):
        tensor = np.zeros(self.neural_net.input_dim)
        for d_type in self.device_types:
            for bus in state.placements[d_type]:
                idx = bus * 4 + self.device_map[d_type]
                tensor[idx] = 1.0
        return tensor