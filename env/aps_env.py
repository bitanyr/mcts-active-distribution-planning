# project_root/env/aps_env.py

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env.state import NetworkState
from optimization.model_builder import build_base_model, evaluate_placement

class ActivePlanningEnv:
    def __init__(self, num_buses=33):
        self.num_buses = num_buses
        self.state = NetworkState(num_buses)
        
        # Scalable penalty for divergent designs (unbearable for the network)
        self.penalty_cost = 10_000_000 
        
        print("Initializing Physics Engine (Pyomo) base structure...")
        self.base_model = build_base_model()

    def reset(self):
        self.state = NetworkState(self.num_buses)
        return self.state

    def step(self, action):
        device_type, bus_index = action

        # 1. Terminal State Evaluation Management (Explicit Stop)
        if device_type == 'stop':
            placement_dict = self.state.get_placement_dict()
            is_feasible, total_cost = evaluate_placement(self.base_model, placement_dict)
            
            if is_feasible:
                return self.state, -total_cost, True, {"msg": "Feasible", "cost": total_cost}
            else:
                return self.state, -self.penalty_cost, True, {"msg": "Infeasible (Voltage/Thermal collapse)", "cost": float('inf')}

        # 2.
        is_valid = self.state.add_device(device_type, bus_index)

        # 3. 
        if not is_valid:
            print(f"       [Env] Capacity limit reached at bus {bus_index}. Forcing terminal evaluation...")
            placement_dict = self.state.get_placement_dict()
            is_feasible, total_cost = evaluate_placement(self.base_model, placement_dict)
            
            if is_feasible:
                return self.state, -total_cost, True, {"msg": "Feasible_MaxCapacity", "cost": total_cost}
            else:
                return self.state, -self.penalty_cost, True, {"msg": "Infeasible_MaxCapacity", "cost": float('inf')}

        # 4. 
        return self.state, 0.0, False, {"msg": "Device placed. Awaiting final design.", "cost": 0.0}

    def base_model_evaluate(self, placement_dict):
        return evaluate_placement(self.base_model, placement_dict)