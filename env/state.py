# project_root/env/state.py

import copy

class NetworkState:
    def __init__(self, num_buses=33, max_devices_per_bus=2, max_total_devices=5):
        self.num_buses = num_buses
        self.max_devices_per_bus = max_devices_per_bus
        self.max_total_devices = max_total_devices 
        
        self.placements = {
            'ess': set(),
            'pv': set(),
            'gas': set(),
            'svc': set(),
            'cb': set()
        }

    def get_node_occupancy(self, bus_index):
        return sum(1 for type_set in self.placements.values() if bus_index in type_set)

    def get_total_installed_devices(self):
        # Calculate the total equipment installed in the entire network
        return sum(len(type_set) for type_set in self.placements.values())

    def add_device(self, device_type, bus_index):
        if self.get_total_installed_devices() >= self.max_total_devices:
            return False

        if self.get_node_occupancy(bus_index) >= self.max_devices_per_bus:
            return False 
            
        if bus_index not in self.placements[device_type]:
            self.placements[device_type].add(bus_index)
            return True 
        return False 

    def get_placement_dict(self):
        return {k: list(v) for k, v in self.placements.items()}

    def get_legal_actions(self):
        legal_actions = []
        
        if self.get_total_installed_devices() >= self.max_total_devices:
            return [('stop', 0)]

        for device in self.placements.keys():
            for bus in range(1, self.num_buses):
                if bus not in self.placements[device] and self.get_node_occupancy(bus) < self.max_devices_per_bus:
                    legal_actions.append((device, bus))
        
        legal_actions.append(('stop', 0))
        return legal_actions

    def clone(self):
        new_state = NetworkState(self.num_buses, self.max_devices_per_bus, self.max_total_devices)
        new_state.placements = copy.deepcopy(self.placements)
        return new_state