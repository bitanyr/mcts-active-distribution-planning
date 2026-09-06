"""Branch-flow, device, market, and soft feasibility constraints."""

import pyomo.environ as pyo

from data.devices import (
    COMPLEMENTARITY_TOL,
    EFF_CH,
    EFF_DIS,
    ESS_INVERTER_OVERSIZE,
    E_ESS_MAX,
    E_ESS_MIN,
    FIXED_PV_NODES,
    GAS_POWER_FACTOR_MARGIN,
    P_ESS_MAX,
    P_GAS_MAX,
    PV_CAPACITY,
    PV_INVERTER_OVERSIZE,
    Q_CB_MAX,
    Q_SVC_MAX,
    S_SUB_MAX,
    V_MAX_SQ,
    V_MIN_SQ,
)
from data.ieee33 import BRANCHES, LOADS
from data.scenarios import LOAD_PROFILE, PV_PROFILE


def define_constraints(model):
    pv_nodes = set(FIXED_PV_NODES)
    scenario_starts = set(int(t) for t in model.T_START)

    # ------------------------------------------------------------------
    # Substation and market exchange
    # ------------------------------------------------------------------

    model.const_sub_v = pyo.Constraint(
        model.T,
        rule=lambda m, t: m.v[0, t] == 1.0,
    )

    model.const_sub_capacity = pyo.Constraint(
        model.T,
        rule=lambda m, t: (
            m.P_sub[t] ** 2 + m.Q_sub[t] ** 2
            <= S_SUB_MAX**2 + m.sub_overload[t]
        ),
    )

    model.const_sub_split = pyo.Constraint(
        model.T,
        rule=lambda m, t: (
            m.P_sub[t] == m.P_sub_import[t] - m.P_sub_export[t]

        ),
    )

    # Prevent artificial simultaneous purchase and sale. Import prices are
    # also strictly above the export tariff, but the explicit condition makes
    # the intended market physics auditable.
    model.const_grid_complementarity = pyo.Constraint(
        model.T,
        rule=lambda m, t: (
            m.P_sub_import[t] * m.P_sub_export[t]
            <= COMPLEMENTARITY_TOL
        ),
    )

    # Export cannot exceed contemporaneous local renewable and gas output.
    model.const_export_cap = pyo.Constraint(
        model.T,
        rule=lambda m, t: (
            m.P_sub_export[t]
            <= sum(
                PV_CAPACITY * PV_PROFILE[int(t)] - m.P_curt_res[i, t]
                for i in pv_nodes
            ) + sum(m.P_gas[i, t] for i in m.N)
        ),
    )

    # ------------------------------------------------------------------
    # Voltage and curtailment
    # ------------------------------------------------------------------

    model.const_v_lower = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, i, t: (
            m.v[i, t] + m.v_viol_down[i, t] >= V_MIN_SQ
        ),
    )
    model.const_v_upper = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, i, t: (
            m.v[i, t] - m.v_viol_up[i, t] <= V_MAX_SQ
        ),
    )

    def res_curtailment_rule(m, i, t):
        available = PV_CAPACITY * PV_PROFILE[int(t)] if i in pv_nodes else 0.0
        return m.P_curt_res[i, t] <= available

    model.const_res_curt = pyo.Constraint(model.N, model.T, rule=res_curtailment_rule)

    model.const_load_curt = pyo.Constraint(

        model.N,
        model.T,
        rule=lambda m, i, t: (
            m.P_curt_aul[i, t]
            <= LOADS[int(i)]["P"] * LOAD_PROFILE[int(t)]
        ),
    )

    # ------------------------------------------------------------------
    # ESS
    # ------------------------------------------------------------------

    model.const_ess_ch_limit = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, i, t: m.P_ch[i, t] <= P_ESS_MAX * m.s_ess[i],
    )
    model.const_ess_dis_limit = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, i, t: m.P_dis[i, t] <= P_ESS_MAX * m.s_ess[i],
    )
    model.const_ess_complementarity = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, i, t: (
            m.P_ch[i, t] * m.P_dis[i, t] <= COMPLEMENTARITY_TOL
        ),
    )

    def soc_update_rule(m, i, t):
        previous = 0.5 * E_ESS_MAX * m.s_ess[i]
        if int(t) not in scenario_starts:
            previous = m.E_soc[i, t - 1]
        return m.E_soc[i, t] == (
            previous
            + EFF_CH * m.P_ch[i, t]
            - m.P_dis[i, t] / EFF_DIS
        )

    model.const_ess_soc_update = pyo.Constraint(model.N, model.T, rule=soc_update_rule)
    model.const_ess_soc_lower = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, i, t: m.E_soc[i, t] >= E_ESS_MIN * m.s_ess[i],
    )
    model.const_ess_soc_upper = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, i, t: m.E_soc[i, t] <= E_ESS_MAX * m.s_ess[i],
    )
    model.const_ess_soc_cyclic = pyo.Constraint(

        model.N,
        model.T_END,
        rule=lambda m, i, t: (
            m.E_soc[i, t] + m.soc_viol_down[i, t] - m.soc_viol_up[i, t]
            == 0.5 * E_ESS_MAX * m.s_ess[i]
        ),
    )

    ess_s_max = ESS_INVERTER_OVERSIZE * P_ESS_MAX
    model.const_ess_inverter = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, i, t: (
            (m.P_dis[i, t] - m.P_ch[i, t]) ** 2 + m.Q_ess[i, t] ** 2
            <= (ess_s_max * m.s_ess[i]) ** 2
        ),
    )

    # ------------------------------------------------------------------
    # PV, gas, SVC, and capacitor bank
    # ------------------------------------------------------------------

    pv_s_max = PV_INVERTER_OVERSIZE * PV_CAPACITY

    def pv_inverter_rule(m, i, t):
        if i not in pv_nodes:
            return m.Q_pv[i, t] == 0.0
        active = PV_CAPACITY * PV_PROFILE[int(t)] - m.P_curt_res[i, t]
        return active**2 + m.Q_pv[i, t] ** 2 <= pv_s_max**2

    model.const_pv_inverter = pyo.Constraint(model.N, model.T, rule=pv_inverter_rule)

    model.const_gas_active = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, i, t: m.P_gas[i, t] <= P_GAS_MAX * m.s_gas[i],
    )
    gas_s_max = GAS_POWER_FACTOR_MARGIN * P_GAS_MAX
    model.const_gas_capability = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, i, t: (
            m.P_gas[i, t] ** 2 + m.Q_gas[i, t] ** 2
            <= (gas_s_max * m.s_gas[i]) ** 2
        ),
    )

    model.const_svc_upper = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, i, t: m.Q_svc[i, t] <= Q_SVC_MAX * m.s_svc[i],
    )

    model.const_svc_lower = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, i, t: m.Q_svc[i, t] >= -Q_SVC_MAX * m.s_svc[i],
    )

    # Aggregated switched-CB relaxation. cb_fraction is continuous because
    # IPOPT is an NLP solver; this modeling assumption is reported explicitly.
    model.const_cb_fraction = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, i, t: m.cb_fraction[i, t] <= m.s_cb[i],
    )
    model.const_cb_voltage = pyo.Constraint(
        model.N,
        model.T,
        rule=lambda m, i, t: (
            m.Q_cb[i, t]
            == Q_CB_MAX * m.cb_fraction[i, t] * m.v[i, t]
        ),
    )

    # ------------------------------------------------------------------
    # DistFlow balances and branch relations
    # ------------------------------------------------------------------

    def active_balance_rule(m, i, t):
        incoming = sum(m.P[k, t] for k in m.E if BRANCHES[int(k)]["to"] == i)
        incoming_loss = sum(
            BRANCHES[int(k)]["r"] * m.l[k, t]
            for k in m.E
            if BRANCHES[int(k)]["to"] == i
        )
        outgoing = sum(m.P[k, t] for k in m.E if BRANCHES[int(k)]["from"] == i)

        pv = 0.0
        if i in pv_nodes:
            pv = PV_CAPACITY * PV_PROFILE[int(t)] - m.P_curt_res[i, t]
        generation = pv + m.P_gas[i, t] + m.P_dis[i, t] - m.P_ch[i, t]
        if i == 0:
            generation += m.P_sub[t]
        served_load = LOADS[int(i)]["P"] * LOAD_PROFILE[int(t)] - m.P_curt_aul[i, t]
        return incoming - incoming_loss + generation == outgoing + served_load

    model.const_p_balance = pyo.Constraint(model.N, model.T, rule=active_balance_rule)

    def reactive_balance_rule(m, i, t):
        incoming = sum(m.Q[k, t] for k in m.E if BRANCHES[int(k)]["to"] == i)
        incoming_loss = sum(
            BRANCHES[int(k)]["x"] * m.l[k, t]
            for k in m.E
            if BRANCHES[int(k)]["to"] == i

        )
        outgoing = sum(m.Q[k, t] for k in m.E if BRANCHES[int(k)]["from"] == i)
        injection = m.Q_gas[i, t] + m.Q_pv[i, t] + m.Q_ess[i, t] + m.Q_svc[i, t] + m.Q_cb[i, t]
        if i == 0:
            injection += m.Q_sub[t]

        p_load = LOADS[int(i)]["P"]
        q_load = LOADS[int(i)]["Q"]
        q_shed = m.P_curt_aul[i, t] * q_load / p_load if p_load > 0.0 else 0.0
        served_q = q_load * LOAD_PROFILE[int(t)] - q_shed
        return incoming - incoming_loss + injection == outgoing + served_q

    model.const_q_balance = pyo.Constraint(model.N, model.T, rule=reactive_balance_rule)

    def voltage_drop_rule(m, k, t):
        branch = BRANCHES[int(k)]
        fb, tb = branch["from"], branch["to"]
        r, x = branch["r"], branch["x"]
        return m.v[tb, t] == (
            m.v[fb, t]
            - 2.0 * (r * m.P[k, t] + x * m.Q[k, t])
            + (r**2 + x**2) * m.l[k, t]
        )

    model.const_v_drop = pyo.Constraint(model.E, model.T, rule=voltage_drop_rule)

    model.const_branch_current_relaxed = pyo.Constraint(
        model.E,
        model.T,
        rule=lambda m, k, t: (
            m.P[k, t] ** 2 + m.Q[k, t] ** 2
            <= m.l[k, t] * m.v[BRANCHES[int(k)]["from"], t]
        ),
    )

    model.const_thermal = pyo.Constraint(
        model.E,
        model.T,
        rule=lambda m, k, t: (
            m.l[k, t] - m.l_viol[k, t]
            <= BRANCHES[int(k)]["i_max_sq"]
        ),
    )

    return model