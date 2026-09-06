"""Independent hard-compliance and SOCP-exactness audit.

Solver termination is not treated as proof that the original physical model
is satisfied. This module separately checks active-constraint residuals,
artificial slacks, raw network limits, complementarity, unserved load,
renewable curtailment, and the branch-flow cone gap.
"""

import math

import pyomo.environ as pyo

from data.devices import (
    COMPLEMENTARITY_TOL,
    CONE_ABS_TOL,
    CONE_REL_TOL,
    CONSTRAINT_TOL,
    E_ESS_MAX,
    FIXED_PV_NODES,
    MAX_LOAD_SHEDDING_MWH_REP,
    MAX_RES_CURTAILMENT_MWH_REP,
    PHYSICAL_TOL,
    PV_CAPACITY,
    SCENARIO_END_HOURS,
    SLACK_TOL,
    S_SUB_MAX,
    V_MAX_SQ,
    V_MIN_SQ,
)
from data.ieee33 import BRANCHES, S_BASE, to_ieee_bus
from data.scenarios import PV_PROFILE, seasonal_weight

SLACK_VARIABLES = (
    "v_viol_down",
    "v_viol_up",
    "sub_overload",
    "l_viol",
    "soc_viol_down",
    "soc_viol_up",
)


def _value(component):
    value = pyo.value(component, exception=False)
    return float("inf") if value is None else float(value)


def _constraint_violation(constraint):
    body = _value(constraint.body)
    if not math.isfinite(body):
        return float("inf")
    violation = 0.0

    if constraint.lower is not None:
        violation = max(violation, _value(constraint.lower) - body)
    if constraint.upper is not None:
        violation = max(violation, body - _value(constraint.upper))
    return max(0.0, violation)


def _collect_constraints(model):
    rows = []
    maximum = 0.0
    worst = None
    for constraint in model.component_data_objects(
        pyo.Constraint, active=True, descend_into=True
    ):
        violation = _constraint_violation(constraint)
        if violation > maximum:
            maximum = violation
            worst = constraint.name
        if violation > 1.0e-12:
            rows.append({"constraint": constraint.name, "violation": violation})
    rows.sort(key=lambda row: row["violation"], reverse=True)
    return {"max": maximum, "worst": worst, "rows": rows}


def _collect_slacks(model):
    rows = []
    summary = {}
    global_max = 0.0
    for name in SLACK_VARIABLES:
        if not hasattr(model, name):
            continue
        variable = getattr(model, name)
        values = []
        for index in variable:
            value = max(0.0, _value(variable[index]))
            values.append(value)
            if value > 1.0e-12:
                rows.append({"slack": name, "index": str(index), "value": value})
        summary[f"{name}_max"] = max(values, default=0.0)
        summary[f"{name}_sum"] = sum(values)
        global_max = max(global_max, summary[f"{name}_max"])
    rows.sort(key=lambda row: row["value"], reverse=True)
    return {"max": global_max, "summary": summary, "rows": rows}


def _collect_physics(model):
    min_v = float("inf")
    max_v = 0.0
    low_v_violation = 0.0
    high_v_violation = 0.0
    thermal_violation = 0.0
    substation_violation = 0.0

    soc_cycle_residual = 0.0
    grid_product_max = 0.0
    ess_product_max = 0.0

    for i in model.N:
        for t in model.T:
            v_sq = _value(model.v[i, t])
            v = math.sqrt(max(0.0, v_sq))
            min_v = min(min_v, v)
            max_v = max(max_v, v)
            low_v_violation = max(low_v_violation, V_MIN_SQ - v_sq)
            high_v_violation = max(high_v_violation, v_sq - V_MAX_SQ)
            ess_product_max = max(
                ess_product_max,
                _value(model.P_ch[i, t]) * _value(model.P_dis[i, t]),
            )

    for k in model.E:
        for t in model.T:
            thermal_violation = max(
                thermal_violation,
                _value(model.l[k, t]) - BRANCHES[int(k)]["i_max_sq"],
            )

    for t in model.T:
        p_sub = _value(model.P_sub[t])
        q_sub = _value(model.Q_sub[t])
        substation_violation = max(
            substation_violation,
            p_sub**2 + q_sub**2 - S_SUB_MAX**2,
        )
        grid_product_max = max(
            grid_product_max,
            _value(model.P_sub_import[t]) * _value(model.P_sub_export[t]),
        )

    for i in model.N:
        target = 0.5 * E_ESS_MAX * _value(model.s_ess[i])
        for t in SCENARIO_END_HOURS:
            soc_cycle_residual = max(
                soc_cycle_residual,
                abs(_value(model.E_soc[i, t]) - target),
            )

    return {
        "min_voltage_pu": min_v,
        "max_voltage_pu": max_v,
        "voltage_lower_violation_sq": max(0.0, low_v_violation),
        "voltage_upper_violation_sq": max(0.0, high_v_violation),
        "thermal_violation_sq": max(0.0, thermal_violation),
        "substation_violation_sq": max(0.0, substation_violation),
        "soc_cycle_residual": max(0.0, soc_cycle_residual),

        "max_grid_import_export_product": max(0.0, grid_product_max),
        "max_ess_charge_discharge_product": max(0.0, ess_product_max),
    }


def _collect_curtailment(model):
    rows = []
    load_rep = 0.0
    res_rep = 0.0
    load_annual = 0.0
    res_annual = 0.0
    available_rep = 0.0
    available_annual = 0.0

    for t in model.T:
        ti = int(t)
        weight = seasonal_weight(ti)
        load_mw = S_BASE * sum(
            max(0.0, _value(model.P_curt_aul[i, t])) for i in model.N
        )
        res_mw = S_BASE * sum(
            max(0.0, _value(model.P_curt_res[i, t])) for i in model.N
        )
        available_mw = (
            S_BASE * len(FIXED_PV_NODES) * PV_CAPACITY * PV_PROFILE[ti]
        )
        load_rep += load_mw
        res_rep += res_mw
        available_rep += available_mw
        load_annual += load_mw * weight
        res_annual += res_mw * weight
        available_annual += available_mw * weight
        rows.append(
            {
                "time": ti,
                "seasonal_weight": weight,
                "load_shedding_mw": load_mw,
                "res_curtailment_mw": res_mw,
                "available_pv_mw": available_mw,
                "load_shedding_mwh_annualized": load_mw * weight,
                "res_curtailment_mwh_annualized": res_mw * weight,
            }
        )

    return {
        "load_shedding_mwh_representative": load_rep,
        "res_curtailment_mwh_representative": res_rep,
        "available_pv_mwh_representative": available_rep,
        "load_shedding_mwh_annualized": load_annual,
        "res_curtailment_mwh_annualized": res_annual,
        "available_pv_mwh_annualized": available_annual,
        "res_curtailment_rate": res_rep / available_rep if available_rep else 0.0,

        "rows": rows,
    }


def _collect_cone(model):
    rows = []
    max_abs = 0.0
    max_rel = 0.0
    nonexact = 0
    for k in model.E:
        branch = BRANCHES[int(k)]
        fb = branch["from"]
        for t in model.T:
            lv = _value(model.l[k, t]) * _value(model.v[fb, t])
            pq = _value(model.P[k, t]) ** 2 + _value(model.Q[k, t]) ** 2
            signed = lv - pq
            absolute = abs(signed)
            relative = absolute / max(abs(lv), abs(pq), 1.0e-8)
            exact = absolute <= CONE_ABS_TOL or relative <= CONE_REL_TOL
            nonexact += int(not exact)
            max_abs = max(max_abs, absolute)
            max_rel = max(max_rel, relative)
            rows.append(
                {
                    "branch_internal": int(k),
                    "from_bus_ieee": to_ieee_bus(fb),
                    "to_bus_ieee": to_ieee_bus(branch["to"]),
                    "time": int(t),
                    "l_times_v": lv,
                    "p2_plus_q2": pq,
                    "signed_gap": signed,
                    "absolute_gap": absolute,
                    "relative_gap": relative,
                    "is_exact": exact,
                }
            )
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "nonexact": nonexact,
        "exact": nonexact == 0,
        "rows": rows,
    }


def audit_compliance(model):
    constraints = _collect_constraints(model)
    slacks = _collect_slacks(model)
    physics = _collect_physics(model)
    curtailment = _collect_curtailment(model)
    cone = _collect_cone(model)

    max_raw = max(
        physics["voltage_lower_violation_sq"],
        physics["voltage_upper_violation_sq"],
        physics["thermal_violation_sq"],
        physics["substation_violation_sq"],
        physics["soc_cycle_residual"],
    )
    complementarity_ok = (
        physics["max_grid_import_export_product"] <= COMPLEMENTARITY_TOL
        and physics["max_ess_charge_discharge_product"] <= COMPLEMENTARITY_TOL
    )
    technical_ok = (
        constraints["max"] <= CONSTRAINT_TOL
        and slacks["max"] <= SLACK_TOL
        and max_raw <= PHYSICAL_TOL
        and complementarity_ok
        and cone["exact"]
    )
    load_ok = (
        curtailment["load_shedding_mwh_representative"]
        <= MAX_LOAD_SHEDDING_MWH_REP
    )
    res_ok = (
        curtailment["res_curtailment_mwh_representative"]
        <= MAX_RES_CURTAILMENT_MWH_REP
    )
    compliant = technical_ok and load_ok and res_ok

    reasons = []
    if constraints["max"] > CONSTRAINT_TOL:
        reasons.append("active constraint residual")
    if slacks["max"] > SLACK_TOL:
        reasons.append("non-zero artificial slack")
    if max_raw > PHYSICAL_TOL:
        reasons.append("raw physical limit violation")
    if not complementarity_ok:
        reasons.append("simultaneous import/export or charge/discharge")
    if not cone["exact"]:
        reasons.append("non-exact branch-flow relaxation")
    if not load_ok:
        reasons.append("load-shedding target not met")
    if not res_ok:
        reasons.append("renewable-curtailment target not met")

    summary = {
        "is_compliant": compliant,
        "technical_compliant": technical_ok,
        "load_target_met": load_ok,
        "res_target_met": res_ok,
        "failure_reasons": "; ".join(reasons),
        "max_constraint_violation": constraints["max"],
        "worst_constraint": constraints["worst"],

        "max_slack": slacks["max"],
        "max_raw_physical_violation": max_raw,
        "max_cone_abs_gap": cone["max_abs"],
        "max_cone_rel_gap": cone["max_rel"],
        "nonexact_cone_points": cone["nonexact"],
        **physics,
        **{k: v for k, v in curtailment.items() if k != "rows"},
        **slacks["summary"],
    }
    return {
        "is_compliant": compliant,
        "summary": summary,
        "slack_details": slacks["rows"],
        "constraint_details": constraints["rows"],
        "curtailment_details": curtailment["rows"],
        "cone_details": cone["rows"],
    }


def failed_compliance_report(reason):
    return {
        "is_compliant": False,
        "summary": {
            "is_compliant": False,
            "technical_compliant": False,
            "failure_reasons": str(reason),
        },
        "slack_details": [],
        "constraint_details": [],
        "curtailment_details": [],
        "cone_details": [],
    }


def export_compliance_to_excel(report, tag, writer):
    """Write auditable summary and detail sheets to an existing writer."""

    import pandas as pd

    tag = str(tag)[:12]
    sheets = (
        ("Compliance", [report["summary"]]),
        ("Slacks", report["slack_details"]),
        ("Curtailment", report["curtailment_details"]),
        ("ConeGap", report["cone_details"]),
        ("Residuals", report["constraint_details"]),
    )
    for suffix, rows in sheets:
        pd.DataFrame(rows).to_excel(
            writer, sheet_name=f"{tag}_{suffix}", index=False
        )