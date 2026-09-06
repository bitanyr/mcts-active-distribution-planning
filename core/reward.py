"""Terminal reward shared by environment, STOP evaluation, and training."""

import math

ECONOMIC_SCALE_RATIO = 0.25


def encode_terminal_reward(
    solver_ok,
    compliance_ok,
    total_economic_cost,
    reference_economic_cost,
):
    """Map a terminal solve to [-1, 1] without changing its ranking.

    Nonconverged and noncompliant designs receive -1. Compliant designs are
    ranked monotonically by the common raw economic objective. A design equal
    to the base case maps to 0.5; cheaper designs map above 0.5.
    """

    reference = float(reference_economic_cost)
    cost = float(total_economic_cost)
    if not math.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_economic_cost must be positive and finite.")
    if not solver_ok or not compliance_ok or not math.isfinite(cost):
        return -1.0
    scale = max(1.0, ECONOMIC_SCALE_RATIO * reference)
    relative_improvement = (reference - cost) / scale
    return float(0.5 + 0.5 * math.tanh(relative_improvement))