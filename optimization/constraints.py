# optimization/ constraints
import pyomo.environ as pyo
from data.ieee33 import BRANCHES, LOADS, T_HORIZON
from data.devices import *
from data.scenarios import LOAD_PROFILE, PV_PROFILE

# Nodes that have solar panels by default
FIXED_PV_NODES = [8, 10, 13, 16, 18, 20, 22, 28]
S_SUB_MAX = 10.0 

# Inverter capacity with 10% oversizing for reactive support
S_PV_INV_CAP = PV_CAPACITY * 1.1
S_ESS_INV_CAP = P_ESS_MAX * 1.1

def define_constraints(model):
    
    
    # Substation Constraints
    def substation_voltage_rule(m, t):
        return m.v[0, t] == 1.0**2
    model.const_sub_v = pyo.Constraint(model.T, rule=substation_voltage_rule)
    
    def substation_capacity_rule(m, t):
        if hasattr(m, 'sub_overload'):
            return m.P_sub[t]**2 + m.Q_sub[t]**2 <= S_SUB_MAX**2 + m.sub_overload[t]
        return m.P_sub[t]**2 + m.Q_sub[t]**2 <= S_SUB_MAX**2
    model.const_sub_capacity = pyo.Constraint(model.T, rule=substation_capacity_rule)
    
    def substation_export_cap_rule(m, t):
        pv_available_now = sum(
            (PV_CAPACITY * PV_PROFILE[t]) - m.P_curt_res[i, t]
            for i in FIXED_PV_NODES
        )
        return m.P_sub[t] >= -pv_available_now
    model.const_sub_export_cap = pyo.Constraint(model.T, rule=substation_export_cap_rule)

    def substation_split_rule(m, t):
        return m.P_sub[t] == m.P_sub_import[t] - m.P_sub_export[t]
    model.const_sub_split = pyo.Constraint(model.T, rule=substation_split_rule)

    # Voltage Limits
    def voltage_lower_rule(m, i, t):
        if hasattr(m, 'v_viol_down'):
            return m.v[i, t] + m.v_viol_down[i, t] >= V_MIN_SQ
        return m.v[i, t] >= V_MIN_SQ
    model.const_v_lower = pyo.Constraint(model.N, model.T, rule=voltage_lower_rule)

    def voltage_upper_rule(m, i, t):
        if hasattr(m, 'v_viol_up'):
            return m.v[i, t] - m.v_viol_up[i, t] <= V_MAX_SQ
        return m.v[i, t] <= V_MAX_SQ
    model.const_v_upper = pyo.Constraint(model.N, model.T, rule=voltage_upper_rule)

    # Energy Storage System (ESS) Constraints
    def ess_charge_limits_rule(m, i, t):
        return m.P_ch[i, t] <= m.s_ess[i] * P_ESS_MAX + 1e-4
    model.const_ess_ch_limits = pyo.Constraint(model.N, model.T, rule=ess_charge_limits_rule)

    def ess_discharge_limits_rule(m, i, t):
        return m.P_dis[i, t] <= m.s_ess[i] * P_ESS_MAX + 1e-4
    model.const_ess_dis_limits = pyo.Constraint(model.N, model.T, rule=ess_discharge_limits_rule)

    def ess_soc_update_rule(m, i, t):
        if t == 0:
            initial_energy = 0.5 * E_ESS_MAX * m.s_ess[i]
            return m.E_soc[i, t] == initial_energy + (m.P_ch[i, t] * EFF_CH) - (m.P_dis[i, t] / EFF_DIS)
        else:
            return m.E_soc[i, t] == m.E_soc[i, t-1] + (m.P_ch[i, t] * EFF_CH) - (m.P_dis[i, t] / EFF_DIS)
    model.const_ess_soc_update = pyo.Constraint(model.N, model.T, rule=ess_soc_update_rule)

    def ess_soc_limits_rule(m, i, t):
        return pyo.inequality(E_ESS_MIN * m.s_ess[i], m.E_soc[i, t], E_ESS_MAX * m.s_ess[i] + 1e-4)
    model.const_ess_soc_limits = pyo.Constraint(model.N, model.T, rule=ess_soc_limits_rule)

    def pv_smart_inverter_rule(m, i, t):
        has_pv = 1 if i in FIXED_PV_NODES else 0
        if has_pv == 1:
            p_pv_actual = (PV_CAPACITY * PV_PROFILE[t]) - m.P_curt_res[i, t]
            return m.Q_pv[i, t]**2 + p_pv_actual**2 <= S_PV_INV_CAP**2 + 1e-4
        else:
            return m.Q_pv[i, t] == 0.0
    model.const_pv_inv = pyo.Constraint(model.N, model.T, rule=pv_smart_inverter_rule)

    def ess_smart_inverter_rule(m, i, t):
        p_ess_active = m.P_ch[i, t] + m.P_dis[i, t]
        return m.Q_ess[i, t]**2 + p_ess_active**2 <= (m.s_ess[i] * S_ESS_INV_CAP)**2 + 1e-4
    model.const_ess_inv = pyo.Constraint(model.N, model.T, rule=ess_smart_inverter_rule)

    # Curtailment and Dispatchable Devices
    def res_curtailment_limit_rule(m, i, t):
        has_pv = 1 if i in FIXED_PV_NODES else 0
        return m.P_curt_res[i, t] <= PV_CAPACITY * PV_PROFILE[t] * has_pv + 1e-4
    model.const_res_curt = pyo.Constraint(model.N, model.T, rule=res_curtailment_limit_rule)

    def load_curtailment_limit_rule(m, i, t):
        return m.P_curt_aul[i, t] <= LOADS[i]['P'] * LOAD_PROFILE[t] + 1e-4
    model.const_load_curt = pyo.Constraint(model.N, model.T, rule=load_curtailment_limit_rule)
    
    def gas_limit_rule(m, i, t):
        return m.P_gas[i, t] <= P_GAS_MAX * m.s_gas[i] + 1e-4
    model.const_gas_limit = pyo.Constraint(model.N, model.T, rule=gas_limit_rule)

    # Soft constraint for battery energy cycle (return to 50%) 
    # If the battery fails to return to 50% charge, the viol variables 
    # are set and penalized in the objective function.
    def ess_soc_soft_cyclic_rule(m, i):
        return m.E_soc[i, m.T.last()] + m.soc_viol_down[i] - m.soc_viol_up[i] == 0.5 * E_ESS_MAX * m.s_ess[i]
    model.const_ess_soc_cyclic = pyo.Constraint(model.N, rule=ess_soc_soft_cyclic_rule)
    
    S_GAS_CAP = P_GAS_MAX * 1.25 
    
    def svc_limit_rule(m, i, t):
        return m.Q_svc[i, t] <= Q_SVC_MAX * m.s_svc[i] + 1e-4
    model.const_svc_limit_up = pyo.Constraint(model.N, model.T, rule=svc_limit_rule)
    
    def svc_limit_down_rule(m, i, t):
        return m.Q_svc[i, t] >= -Q_SVC_MAX * m.s_svc[i] - 1e-4
    model.const_svc_limit_down = pyo.Constraint(model.N, model.T, rule=svc_limit_down_rule)

    def cb_limit_rule(m, i, t):
        return m.Q_cb[i, t] <= Q_CB_MAX * m.s_cb[i] + 1e-4
    model.const_cb_limit = pyo.Constraint(model.N, model.T, rule=cb_limit_rule)

    # DistFlow Branch Equations 

    def active_power_balance_rule(m, i, t):
        p_in = sum(m.P[k, t] for k in m.E if BRANCHES[k]['to'] == i)
        p_loss = sum(BRANCHES[k]['r'] * m.l[k, t] for k in m.E if BRANCHES[k]['to'] == i)
        p_out = sum(m.P[k, t] for k in m.E if BRANCHES[k]['from'] == i)
        
        has_pv = 1 if i in FIXED_PV_NODES else 0
        p_pv = (PV_CAPACITY * PV_PROFILE[t] * has_pv) - m.P_curt_res[i, t]
        p_gas = m.P_gas[i, t]
        p_ess = m.P_dis[i, t] - m.P_ch[i, t]
        p_load = (LOADS[i]['P'] * LOAD_PROFILE[t]) - m.P_curt_aul[i, t]
        
        p_inject = p_pv + p_gas + p_ess
        if i == 0:
            p_inject += m.P_sub[t]
            
        return p_in - p_loss + p_inject == p_out + p_load
    model.const_p_bal = pyo.Constraint(model.N, model.T, rule=active_power_balance_rule)

    def reactive_power_balance_rule(m, i, t):
        q_in = sum(m.Q[k, t] for k in m.E if BRANCHES[k]['to'] == i)
        q_loss = sum(BRANCHES[k]['x'] * m.l[k, t] for k in m.E if BRANCHES[k]['to'] == i)
        q_out = sum(m.Q[k, t] for k in m.E if BRANCHES[k]['from'] == i)
        q_inject = m.Q_svc[i, t] + m.Q_cb[i, t] + m.Q_pv[i, t] + m.Q_ess[i, t] + m.Q_gas[i, t]

        if LOADS[i]['P'] > 0:
            q_curt = m.P_curt_aul[i, t] * (LOADS[i]['Q'] / LOADS[i]['P'])
        else:
            q_curt = 0.0
            
        q_load = (LOADS[i]['Q'] * LOAD_PROFILE[t]) - q_curt
        
        if i == 0:
            q_inject += m.Q_sub[t]
            
        return q_in - q_loss + q_inject == q_out + q_load
    model.const_q_bal = pyo.Constraint(model.N, model.T, rule=reactive_power_balance_rule)

    S_GAS_CAP = P_GAS_MAX * 1.25 

    def gas_capability_curve_rule(m, i, t):

        return m.P_gas[i, t]**2 + m.Q_gas[i, t]**2 <= (S_GAS_CAP * m.s_gas[i])**2 + 1e-4
    model.const_gas_cap_curve = pyo.Constraint(model.N, model.T, rule=gas_capability_curve_rule)

    def gas_active_min_rule(m, i, t):
        P_GAS_MIN = 0 
        return m.P_gas[i, t] >= P_GAS_MIN * m.s_gas[i]
    model.const_gas_min = pyo.Constraint(model.N, model.T, rule=gas_active_min_rule)
    
    def voltage_drop_rule(m, k, t):
        fb = BRANCHES[k]['from']
        tb = BRANCHES[k]['to']
        r = BRANCHES[k]['r']
        x = BRANCHES[k]['x']
        return m.v[tb, t] == m.v[fb, t] - 2*(r*m.P[k, t] + x*m.Q[k, t]) + ((r**2 + x**2)*m.l[k, t])
    model.const_v_drop = pyo.Constraint(model.E, model.T, rule=voltage_drop_rule)

    def branch_current_socp_rule(m, k, t):
        fb = BRANCHES[k]['from']
        return (m.P[k, t]**2 + m.Q[k, t]**2) <= m.l[k, t] * m.v[fb, t]
    model.const_branch_current = pyo.Constraint(model.E, model.T, rule=branch_current_socp_rule)

    def thermal_limit_rule(m, k, t):
        return m.l[k, t] - m.l_viol[k, t] <= I_MAX_SQ
    model.const_thermal = pyo.Constraint(model.E, model.T, rule=thermal_limit_rule)

    return model