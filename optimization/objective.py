# optimization/objective.py
import pyomo.environ as pyo
from data.ieee33 import BRANCHES, S_BASE
from data.devices import *
from data.scenarios import RTP_PRICE

FIXED_PV_NODES = [8, 10, 13, 16, 18, 20, 22, 28]

def get_seasonal_weight(t):
    if t < 24: return 92.0    # بهار
    elif t < 48: return 93.0  # تابستان
    elif t < 72: return 90.0  # پاییز
    else: return 90.0         # زمستان

def define_objective(model):
    def objective_rule(m):
        interest_rate = 0.05
        lifetime = 20
        CRF = (interest_rate * (1 + interest_rate)**lifetime) / (((1 + interest_rate)**lifetime) - 1)
        
        # CAPEX
        inv_cost = sum(
            C_ESS_INV * P_ESS_MAX * m.s_ess[i] +
            C_GAS_INV * P_GAS_MAX * m.s_gas[i] +
            C_SVC_INV * Q_SVC_MAX * m.s_svc[i] +
            C_CB_INV * Q_CB_MAX * m.s_cb[i] +
            (C_PV_INV * PV_CAPACITY if i in FIXED_PV_NODES else 0) 
            for i in m.N
        )
        annual_inv_cost = inv_cost * CRF

        # OPEX
        # Battery thermodynamics (81% efficiency) naturally prevents false arbitrage.
        market_cost = sum(
            RTP_PRICE[t] * (m.P_sub_import[t] - m.P_sub_export[t]) * S_BASE * get_seasonal_weight(t)
            for t in m.T
        )

        degradation_cost = sum(
            C_DEG * (m.P_ch[i, t] + m.P_dis[i, t]) * S_BASE * get_seasonal_weight(t)
            for i in m.N for t in m.T
        )

        curtailment_cost = sum(
            (C_RES * m.P_curt_res[i, t] + C_AUL * m.P_curt_aul[i, t]) * S_BASE * get_seasonal_weight(t)
            for i in m.N for t in m.T
        )

        C_GAS_FUEL = 50.0  
        C_EMISSION = 20.0  
        gas_opex_cost = sum(
            (C_GAS_FUEL + C_EMISSION) * m.P_gas[i, t] * S_BASE * get_seasonal_weight(t)
            for i in m.N for t in m.T
        )

        # Soft Constraints Penalties
        PENALTY_VOLT = 1e7 
        volt_penalty_cost = sum(
            PENALTY_VOLT * (m.v_viol_down[i, t] + m.v_viol_up[i, t]) * get_seasonal_weight(t)
            for i in m.N for t in m.T
        )
        
        PENALTY_OVERLOAD = 1e6
        overload_penalty_cost = sum(
            PENALTY_OVERLOAD * m.sub_overload[t] * get_seasonal_weight(t)
            for t in m.T
        )

        PENALTY_THERMAL = 1e5
        thermal_penalty_cost = sum(
            PENALTY_THERMAL * m.l_viol[k, t] * get_seasonal_weight(t)
            for k in m.E for t in m.T
        )

        PENALTY_SOC = 1e6
        soc_penalty_cost = sum(
            PENALTY_SOC * (m.soc_viol_down[i] + m.soc_viol_up[i]) * 91.25
            for i in m.N
        )

        # SOCP Exactness
        PENALTY_EXACTNESS = 1e-4 
        exactness_penalty_cost = sum(
            PENALTY_EXACTNESS * m.l[k, t] * get_seasonal_weight(t)
            for k in m.E for t in m.T
        )

        return (annual_inv_cost + market_cost + curtailment_cost + gas_opex_cost + 
                degradation_cost + volt_penalty_cost + overload_penalty_cost + 
                thermal_penalty_cost + soc_penalty_cost + exactness_penalty_cost) 

    model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    return model