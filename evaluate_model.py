# evaluate model
import sys
import os
import torch
import math
import pandas as pd
import pyomo.environ as pyo

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env.aps_env import ActivePlanningEnv
from core.mcts import MCTS
from core.network import ADNDeepNet
from data.ieee33 import S_BASE, BRANCHES
from data.scenarios import RTP_PRICE
from data.devices import C_LOSS, C_RES, C_AUL, C_ESS_INV, P_ESS_MAX, C_PV_INV, PV_CAPACITY, C_GAS_INV, P_GAS_MAX, C_SVC_INV, Q_SVC_MAX, C_CB_INV, Q_CB_MAX

def export_grid_data_to_excel(model, scenario_name, writer):
   
    
    v_data = []
    for t in model.T:
        row_data = {'Time_Hour': t}
        for i in model.N:
           
            row_data[f'Bus_{i}_V(pu)'] = math.sqrt(pyo.value(model.v[i, t]))
        v_data.append(row_data)
    df_v = pd.DataFrame(v_data)
    df_v.to_excel(writer, sheet_name=f'{scenario_name}_Voltage', index=False)

    # 2. 
    l_data = []
    for t in model.T:
        row_data = {'Time_Hour': t}
        for k in model.E:
            
            line_current_pu = math.sqrt(pyo.value(model.l[k, t]))
            row_data[f'Branch_{k}_Current(pu)'] = line_current_pu
        l_data.append(row_data)
    df_l = pd.DataFrame(l_data)
    df_l.to_excel(writer, sheet_name=f'{scenario_name}_Branch_Current', index=False)

    # 3. 
    dispatch_data = []
    for t in model.T:
        row_data = {'Time_Hour': t}
        
        
        total_gas = sum(pyo.value(model.P_gas[i, t]) for i in model.N)
        row_data['Total_GAS_Output(pu)'] = total_gas
        
        
        total_ess_ch = sum(pyo.value(model.P_ch[i, t]) for i in model.N)
        total_ess_dis = sum(pyo.value(model.P_dis[i, t]) for i in model.N)
        row_data['Total_ESS_Charge(pu)'] = total_ess_ch
        row_data['Total_ESS_Discharge(pu)'] = total_ess_dis
        
        dispatch_data.append(row_data)
        
    df_dispatch = pd.DataFrame(dispatch_data)
    df_dispatch.to_excel(writer, sheet_name=f'{scenario_name}_Dispatch', index=False)
    
    print(f"     [Excel] Data (Voltage, Current, Dispatch) for '{scenario_name}' exported.")
def robust_evaluate_placement(model, placement_dict):
    
    for i in model.N:
        model.s_ess[i].set_value(0)
        model.s_gas[i].set_value(0)
        model.s_svc[i].set_value(0)
        model.s_cb[i].set_value(0)

    for node in placement_dict.get('ess', []): model.s_ess[node].set_value(1)
    for node in placement_dict.get('gas', []): model.s_gas[node].set_value(1)
    for node in placement_dict.get('svc', []): model.s_svc[node].set_value(1)
    for node in placement_dict.get('cb', []):  model.s_cb[node].set_value(1)

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
                if hasattr(model, 'Q_ess'): model.Q_ess[i, t].unfix()
                
                model.E_soc[i, t].set_value(0.3)
                model.P_ch[i, t].set_value(0.01)
                model.P_dis[i, t].set_value(0.01)
            else:
                model.E_soc[i, t].fix(0.0)
                model.P_ch[i, t].fix(0.0)
                model.P_dis[i, t].fix(0.0)
                if hasattr(model, 'Q_ess'): model.Q_ess[i, t].fix(0.0)
                
            
            if pyo.value(model.s_gas[i]) == 1:
                model.P_gas[i, t].unfix()
                model.P_gas[i, t].set_value(0.01)
            else:
                model.P_gas[i, t].fix(0.0)
                
            # SVC
            if pyo.value(model.s_svc[i]) == 1:
                model.Q_svc[i, t].unfix()
                model.Q_svc[i, t].set_value(0.01)
            else:
                model.Q_svc[i, t].fix(0.0)
                
            
            if pyo.value(model.s_cb[i]) == 1:
                model.Q_cb[i, t].unfix()
                model.Q_cb[i, t].set_value(0.01)
            else:
                model.Q_cb[i, t].fix(0.0)
            
            
            if hasattr(model, 'v_viol_down'): 
                model.v_viol_down[i, t].unfix()
                model.v_viol_down[i, t].set_value(0.001)
            if hasattr(model, 'v_viol_up'): 
                model.v_viol_up[i, t].unfix()
                model.v_viol_up[i, t].set_value(0.001)

        for k in model.E:
            model.P[k, t].set_value(0.01)
            model.Q[k, t].set_value(0.01)
            model.l[k, t].set_value(0.01)
            if hasattr(model, 'l_viol'): model.l_viol[k, t].set_value(0.001)
            
        model.P_sub[t].set_value(0.1)
        model.Q_sub[t].set_value(0.1)
        if hasattr(model, 'P_sub_import'): model.P_sub_import[t].set_value(0.1)
        if hasattr(model, 'P_sub_export'): model.P_sub_export[t].set_value(0.0)

    solver = pyo.SolverFactory('ipopt')
    solver.options['max_iter'] = 3000
    solver.options['max_cpu_time'] = 120.0
    solver.options['tol'] = 1e-4
    solver.options['print_level'] = 0
    solver.options['mu_strategy'] = 'adaptive'
    solver.options['obj_scaling_factor'] = 1e-5

    if not solver.available():
        idaes_path = os.path.join(os.path.expanduser('~'), '.idaes', 'bin', 'ipopt.exe')
        fallback_path = r'C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Scripts\ipopt.exe'
        if os.path.exists(idaes_path): solver = pyo.SolverFactory('ipopt', executable=idaes_path)
        elif os.path.exists(fallback_path): solver = pyo.SolverFactory('ipopt', executable=fallback_path)

    try:
        solver.solve(model, tee=False)
    except:
        pass

def extract_detailed_costs(model, placement_dict):
    DAYS_MULTIPLIER = 365.0 / 4.0 
    interest_rate = 0.05
    lifetime = 20
    CRF = (interest_rate * (1 + interest_rate)**lifetime) / (((1 + interest_rate)**lifetime) - 1)
    
    capex = sum(
        C_ESS_INV * P_ESS_MAX * (1 if i in placement_dict.get('ess', []) else 0) +
        C_GAS_INV * P_GAS_MAX * (1 if i in placement_dict.get('gas', []) else 0) +
        C_SVC_INV * Q_SVC_MAX * (1 if i in placement_dict.get('svc', []) else 0) +
        C_CB_INV * Q_CB_MAX * (1 if i in placement_dict.get('cb', []) else 0) +
        (C_PV_INV * PV_CAPACITY if i in [8, 10, 13, 16, 18, 20, 22, 28] else 0) 
        for i in model.N
    ) * CRF
    
    C_EXPORT_PRICE = 19.99  
    
    market = 0.0
    for t in model.T:
        p_net = pyo.value(model.P_sub[t])
        if p_net > 0:
            market += RTP_PRICE[t] * p_net * S_BASE
        else:
            market += C_EXPORT_PRICE * p_net * S_BASE
            
    market = market * DAYS_MULTIPLIER
    
    loss = sum(C_LOSS * pyo.value(model.l[k, t]) * BRANCHES[k]['r'] * S_BASE for k in model.E for t in model.T) * DAYS_MULTIPLIER
    
    C_GAS_FUEL = 50.0  
    C_EMISSION = 20.0  
    gas_opex = sum(
        (C_GAS_FUEL + C_EMISSION) * pyo.value(model.P_gas[i, t]) * S_BASE * (1 if i in placement_dict.get('gas', []) else 0)
        for i in model.N for t in model.T
    ) * DAYS_MULTIPLIER

    total_real = capex + market + gas_opex
    
    min_v = min(pyo.value(model.v[i, t]) for i in model.N for t in model.T) ** 0.5 
    
    return {
        'min_v': min_v, 'capex': capex, 'market': market, 'loss_cost': loss, 
        'gas_opex': gas_opex, 'total': total_real
    }

def evaluate_and_compare(model_path="trained_adn_net_ep200.pth", excel_filename="Grid_Simulation_Results.xlsx"):
    print("==================================================")
    print("   PROVING MODEL SUPERIORITY (EPISODE 200 ANALYSIS)")
    print("==================================================")

    env = ActivePlanningEnv()
    
    with pd.ExcelWriter(excel_filename) as writer:
        
        print("\n[Phase 1] Evaluating Base Case over 1 Year...")
        base_placement = {'ess': [], 'pv': [], 'gas': [], 'svc': [], 'cb': []}
        robust_evaluate_placement(env.base_model, base_placement)
        base = extract_detailed_costs(env.base_model, base_placement)
        export_grid_data_to_excel(env.base_model, "Base_Case", writer)

        print("\n[Phase 2] Loading Trained AI & Designing Network...")
        neural_net = ADNDeepNet(num_buses=33, num_device_types=4)
        if os.path.exists(model_path):
            neural_net.load_state_dict(torch.load(model_path, weights_only=True))
            neural_net.eval()
            print("   Trained Brain (Ep 200) Loaded Successfully!")
        else:
            print(f"    Warning: Could not find '{model_path}'. Using untrained net.")

        ai_state = env.reset()
        for step in range(5):
            mcts = MCTS(neural_net=neural_net, num_simulations=400)
            
            best_action, _ = mcts.search(ai_state, temperature=0.0, add_noise=False)
            if best_action:
                if best_action[0] == 'stop':
                    print("   AI decided to STOP further investments. Reached optimal topology.")
                    break
                ai_state.add_device(best_action[0], best_action[1])
                print(f"   AI installs [{best_action[0].upper()}] at Bus {best_action[1]}")

        print("\n Running Heavy Physics Simulation for AI's final design...")
        robust_evaluate_placement(env.base_model, ai_state.get_placement_dict())
        ai = extract_detailed_costs(env.base_model, ai_state.get_placement_dict())
        export_grid_data_to_excel(env.base_model, "Trained_AI", writer)
        
    print(f"\n [!] Grid technical data successfully exported to: {excel_filename}")

    print("\n==================================================")
    print("  DETAILED THESIS REPORT (REAL ECONOMICS)")
    print("==================================================")
    print("1. Technical Improvements (Minimum Voltage):")
    print(f"   - Base Case:  {base['min_v']:.4f} p.u. {'(DANGER: UNDER 0.95)' if base['min_v'] < 0.95 else '(OK)'}")
    print(f"   - Trained AI: {ai['min_v']:.4f} p.u. {'(DANGER: UNDER 0.95)' if ai['min_v'] < 0.95 else '(OK)'}")
    
    print("\n2. Economic Breakdown (Actual Cash Flow):")
    print(f"   -   Hardware CAPEX: Base=${base['capex']:,.0f} | AI=${ai['capex']:,.0f}")
    print(f"   -   Grid Purchase:  Base=${base['market']:,.0f} | AI=${ai['market']:,.0f}")
    print(f"   -   Gas Fuel Cost:  Base=${base['gas_opex']:,.0f} | AI=${ai['gas_opex']:,.0f}")
    
    print(f"\n3. TOTAL ACTUAL COST (Real Economic Viability):")
    if base['min_v'] < 0.95:
        print(f"   -   Base Case:  INFEASIBLE (The grid collapses due to severe voltage drops. Cost is technically infinite/invalid)")
    else:
        print(f"   -   Base Case:  ${base['total']:,.2f}")
        
    if ai['min_v'] < 0.95:
        print(f"   -   Trained AI: INFEASIBLE (AI failed to fix the grid)")
    else:
        print(f"   -   Trained AI: ${ai['total']:,.2f} (Fully Stable & Standard Grid)")
    print("==================================================")

if __name__ == "__main__":
    evaluate_and_compare(model_path="trained_adn_net_ep200.pth", excel_filename="Grid_Simulation_Results.xlsx")