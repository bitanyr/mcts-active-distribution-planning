"""Planning environment whose STOP action performs an exact terminal solve."""

from copy import deepcopy

from core.reward import encode_terminal_reward
from env.state import NetworkState
from optimization.model_builder import build_base_model
from optimization.solver import evaluate_placement


class ActivePlanningEnv:
    """Thin environment around the common physics model.

    Intermediate placement actions receive no invented reward.  The only
    learning target is produced when STOP is selected (or when an external
    computational move limit ends an episode), using the same objective and
    compliance audit as final evaluation.
    """

    def __init__(self, model=None, state_factory=NetworkState):
        self.model = model if model is not None else build_base_model()
        self.state_factory = state_factory
        self.state = state_factory()
        self.reference_economic_cost = None
        self._cache = {}
        self.physics_solve_count = 0

    def reset(self):
        self.state = self.state_factory()
        return self.state.clone()

    def set_reference_economic_cost(self, cost):
        value = float(cost)
        if not value > 0.0:
            raise ValueError("Reference economic cost must be positive.")
        self.reference_economic_cost = value
        self._cache.clear()

    @staticmethod
    def _cache_key(state, hard_verify):
        return state.key(), bool(hard_verify)

    def evaluate_terminal_state(
        self,
        state=None,
        *,
        hard_verify=False,
        force_solve=False,
    ):
        """Return a full terminal record; never substitute neural value for STOP."""

        if self.reference_economic_cost is None:

            raise RuntimeError("Set the base-case economic reference before evaluation.")
        terminal_state = self.state if state is None else state
        key = self._cache_key(terminal_state, hard_verify)
        if not force_solve and key in self._cache:
            return deepcopy(self._cache[key])

        self.physics_solve_count += 1
        result = evaluate_placement(
            self.model,
            terminal_state.get_placement_dict(),
            hard_verify=hard_verify,
        )
        result["terminal_value"] = encode_terminal_reward(
            solver_ok=result["solver_ok"],
            compliance_ok=result["is_compliant"],
            total_economic_cost=result["economic_cost"],
            reference_economic_cost=self.reference_economic_cost,
        )
        result["placement_ieee"] = terminal_state.get_ieee_placement_dict()
        self._cache[key] = deepcopy(result)
        return result

    def evaluate_terminal_value(self, state):
        return float(self.evaluate_terminal_state(state)["terminal_value"])

    def step(self, action):
        device, bus = action
        if device == "stop":
            result = self.evaluate_terminal_state(self.state, force_solve=True)
            return self.state.clone(), result["terminal_value"], True, result

        if not self.state.add_device(device, bus):
            info = {
                "accepted": False,
                "failure_reason": f"illegal planning action: {action}",
            }
            return self.state.clone(), -1.0, True, info

        # No arbitrary step reward: the cost/compliance target is terminal.
        return self.state.clone(), 0.0, False, {"accepted": None}