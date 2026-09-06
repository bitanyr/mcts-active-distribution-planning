"""Build the common Pyomo model used by every planning method."""

import pyomo.environ as pyo

from optimization.constraints import define_constraints
from optimization.objective import define_objective
from optimization.variables import define_variables


def build_base_model():
    model = pyo.ConcreteModel(name="Corrected_IEEE33_ADN_Planning")
    define_variables(model)
    define_constraints(model)
    define_objective(model)
    return model