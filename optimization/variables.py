# optimization/ variables
import pyomo.environ as pyo
from data.ieee33 import NUM_BUSES, NUM_BRANCHES, T_HORIZON

def define_variables(model):
    # 1.1
    model.N = pyo.Set(initialize=range(NUM_BUSES))
    model.E = pyo.Set(initialize=range(NUM_BRANCHES))
    model.T = pyo.Set(initialize=range(T_HORIZON))

    # 1.2 
    model.s_ess = pyo.Param(model.N, mutable=True, initialize=0)
    model.s_gas = pyo.Param(model.N, mutable=True, initialize=0)
    model.s_svc = pyo.Param(model.N, mutable=True, initialize=0)
    model.s_cb  = pyo.Param(model.N, mutable=True, initialize=0)

    # 1.3 
    model.P = pyo.Var(model.E, model.T, within=pyo.Reals)
    model.Q = pyo.Var(model.E, model.T, within=pyo.Reals)
    model.l = pyo.Var(model.E, model.T, within=pyo.NonNegativeReals)
    model.v = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    # 1.4
    model.P_sub = pyo.Var(model.T, within=pyo.Reals)
    model.Q_sub = pyo.Var(model.T, within=pyo.Reals)
    model.P_sub_import = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.P_sub_export = pyo.Var(model.T, within=pyo.NonNegativeReals)

    # 1.5 
    model.P_curt_res = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    model.P_curt_aul = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)

    # 1.7
    model.P_ch  = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    model.P_dis = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    model.E_soc = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    model.P_gas = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    
    # Adding a variable reactive power to a gas-fired generator
    model.Q_gas = pyo.Var(model.N, model.T, within=pyo.Reals) 
    model.Q_pv  = pyo.Var(model.N, model.T, within=pyo.Reals)
    model.Q_ess = pyo.Var(model.N, model.T, within=pyo.Reals)
    model.Q_svc = pyo.Var(model.N, model.T, within=pyo.Reals)
    model.Q_cb  = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)

    # 1.6 
    model.v_viol_down = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    model.v_viol_up = pyo.Var(model.N, model.T, within=pyo.NonNegativeReals)
    model.sub_overload = pyo.Var(model.T, within=pyo.NonNegativeReals)
    model.l_viol = pyo.Var(model.E, model.T, within=pyo.NonNegativeReals)

    # Soft constraint variables for battery cycle 
    # (only one variable per node for end of day)
    model.soc_viol_down = pyo.Var(model.N, within=pyo.NonNegativeReals)
    model.soc_viol_up = pyo.Var(model.N, within=pyo.NonNegativeReals)
    return model