# optimization/ model builder
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pyomo.environ as pyo
from optimization.variables import define_variables
from optimization.constraints import define_constraints
from optimization.objective import define_objective

def build_base_model():
    print("Building Base Pyomo Model (This takes a few seconds)...")
    model = pyo.ConcreteModel()
    
    model = define_variables(model)
    model = define_constraints(model)
    model = define_objective(model)
    
    return model

def evaluate_placement(model, placement_dict):
    # 4.1 
    for i in model.N:
        model.s_ess[i].set_value(0)
        model.s_gas[i].set_value(0)
        model.s_svc[i].set_value(0)
        model.s_cb[i].set_value(0)

    for node in placement_dict.get('ess', []): model.s_ess[node].set_value(1)
    for node in placement_dict.get('gas', []): model.s_gas[node].set_value(1)
    for node in placement_dict.get('svc', []): model.s_svc[node].set_value(1)
    for node in placement_dict.get('cb', []):  model.s_cb[node].set_value(1)

    # 4.2 
    for t in model.T:
        for i in model.N:
            
            model.v[i, t].set_value(1.0)
            
            
            model.P_curt_res[i, t].fix(0.0)
            model.P_curt_aul[i, t].fix(0.0)

            
            if hasattr(model, 'Q_pv'):
                model.Q_pv[i, t].fix(0.0)

            
            if pyo.value(model.s_ess[i]) == 1:
                model.E_soc[i, t].unfix()
                model.P_ch[i, t].unfix()
                model.P_dis[i, t].unfix()
                if hasattr(model, 'Q_ess'):
                    model.Q_ess[i, t].unfix() 
                
                model.E_soc[i, t].set_value(0.3) 
                model.P_ch[i, t].set_value(0.01)
                model.P_dis[i, t].set_value(0.01)
                if hasattr(model, 'Q_ess'):
                    model.Q_ess[i, t].set_value(0.0)
            else:
                model.E_soc[i, t].fix(0.0)
                model.P_ch[i, t].fix(0.0)
                model.P_dis[i, t].fix(0.0)
                if hasattr(model, 'Q_ess'):
                    model.Q_ess[i, t].fix(0.0) 
            
            
            if pyo.value(model.s_gas[i]) == 1:
                model.P_gas[i, t].unfix()
                model.P_gas[i, t].set_value(0.01)

                model.Q_gas[i, t].unfix()
                model.Q_gas[i, t].set_value(0.01)
            else:
                model.P_gas[i, t].fix(0.0)
                model.Q_gas[i, t].fix(0.0)
            
            if pyo.value(model.s_cb[i]) == 1:
                model.Q_cb[i, t].unfix()
                model.Q_cb[i, t].set_value(0.01)
            else:
                model.Q_cb[i, t].fix(0.0)

            
            if pyo.value(model.s_svc[i]) == 1:
                model.Q_svc[i, t].unfix()
                model.Q_svc[i, t].set_value(0.01)
            else:
                model.Q_svc[i, t].fix(0.0)
            
            
            if hasattr(model, 'v_viol_down'): 
                model.v_viol_down[i, t].unfix()
                model.v_viol_down[i, t].set_value(0.001)
            if hasattr(model, 'v_viol_up'): 
                model.v_viol_up[i, t].unfix()
                model.v_viol_up[i, t].set_value(0.001)

        
        if hasattr(model, 'soc_viol_down'): model.soc_viol_down[i].set_value(0.001)
        if hasattr(model, 'soc_viol_up'): model.soc_viol_up[i].set_value(0.001)
        
        for k in model.E:
            model.P[k, t].set_value(0.01)
            model.Q[k, t].set_value(0.01)
            model.l[k, t].set_value(0.01)
            if hasattr(model, 'l_viol'): 
                model.l_viol[k, t].set_value(0.001)
        
        model.P_sub[t].set_value(0.1)
        model.Q_sub[t].set_value(0.1)
        if hasattr(model, 'P_sub_import'): model.P_sub_import[t].set_value(0.1)
        if hasattr(model, 'P_sub_export'): model.P_sub_export[t].set_value(0.0)
        
        if hasattr(model, 'sub_overload'): 
            model.sub_overload[t].set_value(0.001)

    # 4.3 
    solver = pyo.SolverFactory('ipopt')
    solver.options['warm_start_init_point'] = 'yes'
    solver.options['max_iter'] = 3000
    solver.options['max_cpu_time'] = 120.0 
    solver.options['tol'] = 1e-4
    solver.options['print_level'] = 0
    solver.options['mu_strategy'] = 'adaptive'
    solver.options['obj_scaling_factor'] = 1e-5
    solver.options['bound_push'] = 1e-6

    if not solver.available():
        idaes_path = os.path.join(os.path.expanduser('~'), '.idaes', 'bin', 'ipopt.exe')
        if os.path.exists(idaes_path):
            solver = pyo.SolverFactory('ipopt', executable=idaes_path)
        else:
            fallback_path = r'C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Scripts\ipopt.exe'
            if os.path.exists(fallback_path):
                solver = pyo.SolverFactory('ipopt', executable=fallback_path)
            else:
                 raise RuntimeError(f"Cannot find ipopt.exe in {idaes_path} or {fallback_path}")

    # 4.4
    try:
        result = solver.solve(model, tee=False)
        if (result.solver.status == pyo.SolverStatus.ok) and \
           (result.solver.termination_condition == pyo.TerminationCondition.optimal):
            total_cost = pyo.value(model.obj)
            return True, total_cost
        else:
            return False, float('inf')
    except Exception as e:
        return False, float('inf')

if __name__ == "__main__":
    my_model = build_base_model()
    print("\n--- Test 1: No Extra Devices (Only Fixed PVs) ---")
    is_feasible, cost = evaluate_placement(my_model, {})
    if is_feasible:
        print(f"Network is Feasible. Total Cost: ${cost:,.2f}")
    else:
        print("Network is INFEASIBLE.")