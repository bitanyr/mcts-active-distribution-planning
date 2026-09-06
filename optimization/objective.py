"""Single source of truth for optimization and financial reporting."""

import pyomo.environ as pyo

from data.devices import (
    C_AUL,
    C_CB_INV,
    C_DEG,
    C_EMISSION,
    C_ESS_INV,
    C_EXPORT_PRICE,
    C_GAS_FUEL,
    C_GAS_INV,
    C_PV_INV,
    C_RES,
    C_SVC_INV,
    EXACTNESS_REGULARIZATION,
    FIXED_PV_NODES,
    INCLUDE_FIXED_PV_CAPEX,
    PENALTY_OVERLOAD,
    PENALTY_SOC,
    PENALTY_THERMAL,
    PENALTY_VOLT,
    P_ESS_MAX,
    P_GAS_MAX,
    PV_CAPACITY,
    Q_CB_MAX,
    Q_SVC_MAX,
    capital_recovery_factor,
)
from data.ieee33 import BRANCHES, S_BASE
from data.scenarios import RTP_PRICE, seasonal_weight


def define_objective(model):
    crf = capital_recovery_factor()

    model.incremental_capex = pyo.Expression(
        expr=sum(
            C_ESS_INV * P_ESS_MAX * model.s_ess[i]
            + C_GAS_INV * P_GAS_MAX * model.s_gas[i]
            + C_SVC_INV * Q_SVC_MAX * model.s_svc[i]
            + C_CB_INV * Q_CB_MAX * model.s_cb[i]
            for i in model.N
        )
    )

    fixed_pv_capex = C_PV_INV * PV_CAPACITY * len(FIXED_PV_NODES)
    model.fixed_pv_capex = pyo.Expression(
        expr=fixed_pv_capex if INCLUDE_FIXED_PV_CAPEX else 0.0
    )
    model.cost_annualized_investment = pyo.Expression(

        expr=crf * (model.incremental_capex + model.fixed_pv_capex)
    )

    model.cost_market = pyo.Expression(
        expr=sum(
            (
                RTP_PRICE[int(t)] * model.P_sub_import[t]
                - C_EXPORT_PRICE * model.P_sub_export[t]
            )
            * S_BASE
            * seasonal_weight(t)
            for t in model.T
        )
    )

    model.cost_gas_opex = pyo.Expression(
        expr=sum(
            (C_GAS_FUEL + C_EMISSION)
            * model.P_gas[i, t]
            * S_BASE
            * seasonal_weight(t)
            for i in model.N
            for t in model.T
        )
    )
    model.cost_ess_degradation = pyo.Expression(
        expr=sum(
            C_DEG
            * (model.P_ch[i, t] + model.P_dis[i, t])
            * S_BASE
            * seasonal_weight(t)
            for i in model.N
            for t in model.T
        )
    )
    model.cost_res_curtailment = pyo.Expression(
        expr=sum(
            C_RES * model.P_curt_res[i, t] * S_BASE * seasonal_weight(t)
            for i in model.N
            for t in model.T
        )
    )
    model.cost_load_shedding = pyo.Expression(
        expr=sum(
            C_AUL * model.P_curt_aul[i, t] * S_BASE * seasonal_weight(t)
            for i in model.N
            for t in model.T
        )
    )

    # Losses already increase upstream purchases. Reporting them again as an
    # energy metric is useful, but monetizing them again would double count.

    model.annual_loss_energy_mwh = pyo.Expression(
        expr=sum(
            BRANCHES[int(k)]["r"]
            * model.l[k, t]
            * S_BASE
            * seasonal_weight(t)
            for k in model.E
            for t in model.T
        )
    )

    model.cost_voltage_penalty = pyo.Expression(
        expr=sum(
            PENALTY_VOLT
            * (model.v_viol_down[i, t] + model.v_viol_up[i, t])
            * seasonal_weight(t)
            for i in model.N
            for t in model.T
        )
    )
    model.cost_substation_penalty = pyo.Expression(
        expr=sum(
            PENALTY_OVERLOAD * model.sub_overload[t] * seasonal_weight(t)
            for t in model.T
        )
    )
    model.cost_thermal_penalty = pyo.Expression(
        expr=sum(
            PENALTY_THERMAL * model.l_viol[k, t] * seasonal_weight(t)
            for k in model.E
            for t in model.T
        )
    )
    model.cost_soc_penalty = pyo.Expression(
        expr=sum(
            PENALTY_SOC
            * (model.soc_viol_down[i, t] + model.soc_viol_up[i, t])
            * 0.25 * 365.0
            for i in model.N
            for t in model.T_END
        )
    )

    model.cost_exactness_regularization = pyo.Expression(
        expr=sum(
            EXACTNESS_REGULARIZATION * model.l[k, t] * seasonal_weight(t)
            for k in model.E
            for t in model.T
        )
    )

    model.cost_total_economic = pyo.Expression(

        expr=(
            model.cost_annualized_investment
            + model.cost_market
            + model.cost_gas_opex
            + model.cost_ess_degradation
            + model.cost_res_curtailment
            + model.cost_load_shedding
        )
    )
    model.cost_total_penalty = pyo.Expression(
        expr=(
            model.cost_voltage_penalty
            + model.cost_substation_penalty
            + model.cost_thermal_penalty
            + model.cost_soc_penalty
        )
    )
    model.cost_total_optimization = pyo.Expression(
        expr=(
            model.cost_total_economic
            + model.cost_total_penalty
            + model.cost_exactness_regularization
        )
    )
    model.obj = pyo.Objective(expr=model.cost_total_optimization, sense=pyo.minimize)
    return model