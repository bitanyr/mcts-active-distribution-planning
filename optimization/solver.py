"""One placement configuration, one IPOPT configuration, one cost report."""

import os
import shutil
from contextlib import contextmanager, nullcontext
from pathlib import Path

import pyomo.environ as pyo

from data.devices import DEVICE_TYPES
from data.ieee33 import BRANCHES, NUM_BUSES
from optimization.compliance import audit_compliance, failed_compliance_report

SLACK_NAMES = (
    "v_viol_down",
    "v_viol_up",
    "sub_overload",
    "l_viol",
    "soc_viol_down",
    "soc_viol_up",
)


def _set_value(variable, value):
    variable.set_value(value, skip_validation=True)


def _set_fixed(variable, installed, initial=0.0):
    if installed:
        variable.unfix()
        _set_value(variable, initial)
    else:
        variable.fix(0.0)


def normalize_placement(placement):
    """Validate a placement and return sorted unique internal bus indices."""

    unknown = set(placement) - set(DEVICE_TYPES) - {"pv"}
    if unknown:
        raise ValueError(f"Unknown placement keys: {sorted(unknown)}")

    normalized = {}
    for device in DEVICE_TYPES:
        buses = sorted(set(int(bus) for bus in placement.get(device, ())))
        for bus in buses:
            if not 1 <= bus < NUM_BUSES:
                raise ValueError(
                    f"{device} installation bus must be an internal index in "
                    f"[1, {NUM_BUSES - 1}], received {bus}."
                )
        normalized[device] = buses

    return normalized


def configure_placement(model, placement):
    """Apply placement parameters and identical operating freedom for all calls."""

    placement = normalize_placement(placement)
    installed = {device: set(buses) for device, buses in placement.items()}

    for i in model.N:
        ii = int(i)
        model.s_ess[i] = float(ii in installed["ess"])
        model.s_gas[i] = float(ii in installed["gas"])
        model.s_svc[i] = float(ii in installed["svc"])
        model.s_cb[i] = float(ii in installed["cb"])

    for i in model.N:
        ii = int(i)
        for t in model.T:
            _set_value(model.v[i, t], 1.0)
            model.P_curt_res[i, t].unfix()
            model.P_curt_aul[i, t].unfix()
            _set_value(model.P_curt_res[i, t], 0.0)
            _set_value(model.P_curt_aul[i, t], 0.0)

            # PV reactive support is available in every evaluation, including
            # training, final evaluation, and all baselines.
            if ii in (8, 10, 13, 16, 18, 20, 22, 28):
                model.Q_pv[i, t].unfix()
                _set_value(model.Q_pv[i, t], 0.0)
            else:
                model.Q_pv[i, t].fix(0.0)

            _set_fixed(model.P_ch[i, t], ii in installed["ess"], 0.0)
            _set_fixed(model.P_dis[i, t], ii in installed["ess"], 0.0)
            _set_fixed(model.E_soc[i, t], ii in installed["ess"], 0.25)
            _set_fixed(model.Q_ess[i, t], ii in installed["ess"], 0.0)

            _set_fixed(model.P_gas[i, t], ii in installed["gas"], 0.0)
            _set_fixed(model.Q_gas[i, t], ii in installed["gas"], 0.0)
            _set_fixed(model.Q_svc[i, t], ii in installed["svc"], 0.0)
            _set_fixed(model.cb_fraction[i, t], ii in installed["cb"], 0.0)
            _set_fixed(model.Q_cb[i, t], ii in installed["cb"], 0.0)

            model.v_viol_down[i, t].unfix()
            model.v_viol_up[i, t].unfix()
            _set_value(model.v_viol_down[i, t], 0.0)
            _set_value(model.v_viol_up[i, t], 0.0)

        for t in model.T_END:
            model.soc_viol_down[i, t].unfix()
            model.soc_viol_up[i, t].unfix()

            _set_value(model.soc_viol_down[i, t], 0.0)
            _set_value(model.soc_viol_up[i, t], 0.0)

    for k in model.E:
        for t in model.T:
            _set_value(model.P[k, t], 0.0)
            _set_value(model.Q[k, t], 0.0)
            _set_value(model.l[k, t], 0.01)
            model.l_viol[k, t].unfix()
            _set_value(model.l_viol[k, t], 0.0)

    for t in model.T:
        for variable, initial in (
            (model.P_sub[t], 0.1),
            (model.Q_sub[t], 0.0),
            (model.P_sub_import[t], 0.1),
            (model.P_sub_export[t], 0.0),
        ):
            variable.unfix()
            _set_value(variable, initial)
        model.sub_overload[t].unfix()
        _set_value(model.sub_overload[t], 0.0)

    return placement


@contextmanager
def hard_physics_mode(model):
    """Run zero-slack verification with independent cone-gap auditing.

    The relaxed branch-current constraint remains active. A solution is
    accepted only if the compliance audit confirms that the relaxation is
    numerically exact.
    """

    fixed_state = []

    for name in SLACK_NAMES:
        variable = getattr(model, name, None)

        if variable is None:
            continue

        for index in variable:
            data = variable[index]

            fixed_state.append(
                (
                    data,
                    data.fixed,
                    pyo.value(data, exception=False),
                )
            )

            data.fix(0.0)

    try:
        yield

    finally:
        for data, was_fixed, old_value in fixed_state:
            if was_fixed:
                data.fix(old_value)
            else:
                data.unfix()
                if old_value is not None:
                    _set_value(data, old_value)


def _find_ipopt_executable():
    explicit = os.environ.get("IPOPT_EXECUTABLE")
    candidates = [explicit, shutil.which("ipopt")]
    candidates.extend(
        (
            str(Path.home() / ".idaes" / "bin" / "ipopt"),
            str(Path.home() / ".idaes" / "bin" / "ipopt.exe"),
        )
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate))
    return None


def build_ipopt_solver():
    executable = _find_ipopt_executable()
    solver = (
        pyo.SolverFactory("ipopt", executable=executable)
        if executable
        else pyo.SolverFactory("ipopt")
    )
    if not solver.available(exception_flag=False):
        raise RuntimeError(
            "IPOPT is not available. Install IPOPT on PATH or set the "
            "IPOPT_EXECUTABLE environment variable to its full path."
        )
    solver.options.update(
        {
            "tol": 1.0e-7,
            "constr_viol_tol": 1.0e-7,
            "bound_relax_factor": 0.0,
            "honor_original_bounds": "yes",

            "acceptable_tol": 1.0e-6,
            "max_iter": 5000,
            "max_cpu_time": 300.0,
            "mu_strategy": "adaptive",
            "warm_start_init_point": "yes",
            "nlp_scaling_method": "gradient-based",
            "print_level": 0,
        }
    )
    return solver


def _solver_succeeded(result):
    return (
        result.solver.status == pyo.SolverStatus.ok
        and result.solver.termination_condition
        in (
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
        )
    )


def get_cost_breakdown(model):
    """Read every financial number from the objective's Pyomo expressions."""

    costs = {
        "incremental_capex": pyo.value(model.incremental_capex),
        "fixed_pv_capex": pyo.value(model.fixed_pv_capex),
        "annualized_investment": pyo.value(model.cost_annualized_investment),
        "market": pyo.value(model.cost_market),
        "gas_opex": pyo.value(model.cost_gas_opex),
        "ess_degradation": pyo.value(model.cost_ess_degradation),
        "res_curtailment": pyo.value(model.cost_res_curtailment),
        "load_shedding": pyo.value(model.cost_load_shedding),
        "voltage_penalty": pyo.value(model.cost_voltage_penalty),
        "substation_penalty": pyo.value(model.cost_substation_penalty),
        "thermal_penalty": pyo.value(model.cost_thermal_penalty),
        "soc_penalty": pyo.value(model.cost_soc_penalty),
        "total_economic": pyo.value(model.cost_total_economic),
        "total_penalty": pyo.value(model.cost_total_penalty),
        "exactness_regularization": pyo.value(model.cost_exactness_regularization),
        "total_objective": pyo.value(model.cost_total_optimization),
        "annual_loss_energy_mwh": pyo.value(model.annual_loss_energy_mwh),
    }
    expected = (
        costs["total_economic"]
        + costs["total_penalty"]
        + costs["exactness_regularization"]
    )
    if abs(costs["total_objective"] - expected) > 1.0e-3:
        raise RuntimeError("Objective and cost breakdown are inconsistent.")

    return {key: float(value) for key, value in costs.items()}


def evaluate_placement(model, placement, hard_verify=False, tee=False):
    """Solve and audit one placement using a base-case warm start.

    The base case is solved only once for each model instance. Its
    operating point is then retained as the initial point for the
    first candidate placement.
    """

    normalized = configure_placement(model, placement)
    context = hard_physics_mode(model) if hard_verify else nullcontext()

    solver_ok = False
    result = None
    compliance = failed_compliance_report("solve not started")
    costs = None

    try:
        solver = build_ipopt_solver()

        # Build a physically meaningful initial point only once for
        # each model instance.
        if not getattr(model, "_base_warm_started", False):
            configure_placement(model, {})

            base_result = solver.solve(model, tee=tee)

            if not _solver_succeeded(base_result):
                result = base_result
                compliance = failed_compliance_report(
                    "base-case warm start failed: "
                    f"{base_result.solver.termination_condition}"
                )
            else:
                object.__setattr__(
                    model,
                    "_base_warm_started",
                    True,
                )

        if getattr(model, "_base_warm_started", False):
            # Restore the requested placement while retaining the
            # solved network variables as the IPOPT initial point.
            configure_placement(model, normalized)

            with context:
                result = solver.solve(model, tee=tee)
                solver_ok = _solver_succeeded(result)

                if solver_ok:
                    compliance = audit_compliance(model)
                    costs = get_cost_breakdown(model)
                else:
                    compliance = failed_compliance_report(
                        result.solver.termination_condition
                    )

    except Exception as exc:
        compliance = failed_compliance_report(
            f"solver exception: {exc}"
        )

    accepted = bool(
        solver_ok and compliance["is_compliant"]
    )

    evaluation = {
        "solver_ok": bool(solver_ok),
        "is_compliant": bool(compliance["is_compliant"]),
        "accepted": accepted,
        "hard_verified": bool(hard_verify and accepted),
        "placement": normalized,
        "objective_cost": (
            costs["total_objective"]
            if costs is not None
            else float("inf")
        ),
        "economic_cost": (
            costs["total_economic"]
            if costs is not None
            else float("inf")
        ),
        "penalty_cost": (
            costs["total_penalty"]
            if costs is not None
            else float("inf")
        ),
        "costs": costs,
        "compliance": compliance,
        "termination_condition": (
            str(result.solver.termination_condition)
            if result is not None
            else None
        ),
    }

    model.last_evaluation = evaluation
    return evaluation