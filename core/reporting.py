"""Tabular exports generated from the solved common Pyomo model."""

import math

import pandas as pd
import pyomo.environ as pyo

from data.ieee33 import BRANCHES, I_BASE_KA, S_BASE, to_ieee_bus


def _value(component):
    value = pyo.value(component, exception=False)
    return float("nan") if value is None else float(value)


def voltage_rows(model, method):
    return [
        {
            "method": method,
            "bus_ieee": to_ieee_bus(i),
            "time": int(t),
            "voltage_pu": math.sqrt(max(0.0, _value(model.v[i, t]))),
        }
        for i in model.N
        for t in model.T
    ]


def current_rows(model, method):
    rows = []
    for k in model.E:
        branch = BRANCHES[int(k)]
        for t in model.T:
            current_pu = math.sqrt(max(0.0, _value(model.l[k, t])))
            rows.append(
                {
                    "method": method,
                    "branch": int(k) + 1,
                    "from_bus_ieee": to_ieee_bus(branch["from"]),
                    "to_bus_ieee": to_ieee_bus(branch["to"]),
                    "time": int(t),
                    "current_pu": current_pu,
                    "current_ka": current_pu * I_BASE_KA,
                    "rating_mva": branch["rating_mva"],
                }
            )
    return rows


def dispatch_rows(model, method):
    rows = []
    for t in model.T:

        rows.append(
            {
                "method": method,
                "time": int(t),
                "grid_import_mw": S_BASE * _value(model.P_sub_import[t]),
                "grid_export_mw": S_BASE * _value(model.P_sub_export[t]),
                "gas_mw": S_BASE * sum(_value(model.P_gas[i, t]) for i in model.N),
                "ess_charge_mw": S_BASE * sum(_value(model.P_ch[i, t]) for i in model.N),
                "ess_discharge_mw": S_BASE * sum(_value(model.P_dis[i, t]) for i in model.N),
                "load_shedding_mw": S_BASE * sum(_value(model.P_curt_aul[i, t]) for i in model.N),
                "renewable_curtailment_mw": S_BASE * sum(_value(model.P_curt_res[i, t]) for i in model.N),
            }
        )
    return rows


def write_solution_snapshot(writer, model, method):
    prefix = str(method)[:12]
    pd.DataFrame(voltage_rows(model, method)).to_excel(
        writer, sheet_name=f"{prefix}_Voltage", index=False
    )
    pd.DataFrame(current_rows(model, method)).to_excel(
        writer, sheet_name=f"{prefix}_Current", index=False
    )
    pd.DataFrame(dispatch_rows(model, method)).to_excel(
        writer, sheet_name=f"{prefix}_Dispatch", index=False
    )


def flatten_summary(method, result):
    row = {
        "method": method,
        "solver_ok": result["solver_ok"],
        "compliant": result["is_compliant"],
        "hard_verified": result["hard_verified"],
        "economic_cost_usd_per_year": result["economic_cost"],
        "optimization_objective": result["objective_cost"],
        "penalty_cost": result["penalty_cost"],
        "termination_condition": result["termination_condition"],
        "placement_internal": str(result["placement"]),
        "placement_ieee": str(result.get("placement_ieee", {})),
    }
    row.update(result["compliance"].get("summary", {}))
    if result.get("costs"):
        row.update({f"cost_{key}": value for key, value in result["costs"].items()})
    return row