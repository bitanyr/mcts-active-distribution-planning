"""Canonical immutable-by-cloning planning state."""

from copy import deepcopy

from data.devices import DEVICE_TYPES
from data.ieee33 import NUM_BUSES, to_ieee_bus


class NetworkState:
    """Set-valued installation state with no arbitrary five-device cap."""

    def __init__(
        self,
        num_buses=NUM_BUSES,
        max_devices_per_bus=None,
        max_total_devices=None,
    ):
        self.num_buses = int(num_buses)
        self.device_types = tuple(DEVICE_TYPES)
        self.max_devices_per_bus = (
            len(self.device_types)
            if max_devices_per_bus is None
            else int(max_devices_per_bus)
        )
        self.max_total_devices = (
            len(self.device_types) * (self.num_buses - 1)
            if max_total_devices is None
            else int(max_total_devices)
        )
        self.placements = {device: set() for device in self.device_types}

    def get_node_occupancy(self, bus_index):
        return sum(bus_index in buses for buses in self.placements.values())

    def get_total_installed_devices(self):
        return sum(len(buses) for buses in self.placements.values())

    def add_device(self, device_type, bus_index):
        if device_type not in self.placements:
            return False
        bus = int(bus_index)
        if not 1 <= bus < self.num_buses:
            return False
        if self.get_total_installed_devices() >= self.max_total_devices:
            return False
        if self.get_node_occupancy(bus) >= self.max_devices_per_bus:
            return False
        if bus in self.placements[device_type]:
            return False
        self.placements[device_type].add(bus)
        return True

    def get_legal_actions(self):
        actions = []
        if self.get_total_installed_devices() < self.max_total_devices:
            for bus in range(1, self.num_buses):
                if self.get_node_occupancy(bus) >= self.max_devices_per_bus:
                    continue
                for device in self.device_types:
                    if bus not in self.placements[device]:
                        actions.append((device, bus))
        actions.append(("stop", 0))
        return actions

    def get_placement_dict(self):
        return {
            device: sorted(int(bus) for bus in buses)
            for device, buses in self.placements.items()
        }

    def get_ieee_placement_dict(self):
        return {
            device: [to_ieee_bus(bus) for bus in buses]
            for device, buses in self.get_placement_dict().items()
        }

    def key(self):
        return tuple(
            (device, tuple(sorted(self.placements[device])))
            for device in self.device_types
        )

    def clone(self):
        clone = NetworkState(
            num_buses=self.num_buses,
            max_devices_per_bus=self.max_devices_per_bus,
            max_total_devices=self.max_total_devices,
        )
        clone.placements = deepcopy(self.placements)
        return clone