"""Pyomo sets, planning parameters, and operating variables."""

import pyomo.environ as pyo

from data.devices import SCENARIO_END_HOURS, SCENARIO_START_HOURS
from data.ieee33 import NUM_BRANCHES, NUM_BUSES, T_HORIZON


def define_variables(model):
    model.N = pyo.RangeSet(0, NUM_BUSES - 1)
    model.E = pyo.RangeSet(0, NUM_BRANCHES - 1)
    model.T = pyo.RangeSet(0, T_HORIZON - 1)
    model.T_START = pyo.Set(initialize=SCENARIO_START_HOURS, ordered=True)
    model.T_END = pyo.Set(initialize=SCENARIO_END_HOURS, ordered=True)

    # Fixed placement parameters changed by configure_placement().
    model.s_ess = pyo.Param(model.N, mutable=True, initialize=0.0)
    model.s_gas = pyo.Param(model.N, mutable=True, initialize=0.0)
    model.s_svc = pyo.Param(model.N, mutable=True, initialize=0.0)
    model.s_cb = pyo.Param(model.N, mutable=True, initialize=0.0)

    # Branch-flow variables.
    model.P = pyo.Var(model.E, model.T, domain=pyo.Reals, initialize=0.0)
    model.Q = pyo.Var(model.E, model.T, domain=pyo.Reals, initialize=0.0)
    model.l = pyo.Var(model.E, model.T, domain=pyo.NonNegativeReals, initialize=0.01)
    model.v = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals, initialize=1.0)

    # Upstream grid exchange.
    model.P_sub = pyo.Var(model.T, domain=pyo.Reals, initialize=0.1)
    model.Q_sub = pyo.Var(model.T, domain=pyo.Reals, initialize=0.0)
    model.P_sub_import = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=0.1)
    model.P_sub_export = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=0.0)

    # Curtailment and unserved load.
    model.P_curt_res = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.P_curt_aul = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0)

    # ESS dispatch.
    model.P_ch = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.P_dis = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.E_soc = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.Q_ess = pyo.Var(model.N, model.T, domain=pyo.Reals, initialize=0.0)

    # Other technologies.
    model.P_gas = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.Q_gas = pyo.Var(model.N, model.T, domain=pyo.Reals, initialize=0.0)
    model.Q_pv = pyo.Var(model.N, model.T, domain=pyo.Reals, initialize=0.0)
    model.Q_svc = pyo.Var(model.N, model.T, domain=pyo.Reals, initialize=0.0)
    model.cb_fraction = pyo.Var(model.N, model.T, bounds=(0.0, 1.0), initialize=0.0)
    model.Q_cb = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0)

    # Soft slacks are used for informative search. Final verification fixes all

    # of them to zero and restores the exact branch-current equality.
    model.v_viol_down = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.v_viol_up = pyo.Var(model.N, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.sub_overload = pyo.Var(model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.l_viol = pyo.Var(model.E, model.T, domain=pyo.NonNegativeReals, initialize=0.0)
    model.soc_viol_down = pyo.Var(model.N, model.T_END, domain=pyo.NonNegativeReals, initialize=0.0)
    model.soc_viol_up = pyo.Var(model.N, model.T_END, domain=pyo.NonNegativeReals, initialize=0.0)

    return model