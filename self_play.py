# self_play.py
import sys
import os
import torch
import logging
import math
import pandas as pd
import torch.optim as optim
import pyomo.environ as pyo

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from env.aps_env import ActivePlanningEnv
from core.mcts import MCTS
from core.network import ADNDeepNet
from core.replay_buffer import ReplayBuffer

logging.getLogger('pyomo.core').setLevel(logging.ERROR)

def verify_exact_physics(model):
    
    try:
        max_gap = 0.0
        from data.ieee33 import BRANCHES
        for k in model.E:
            for t in model.T:
                fb = BRANCHES[k]['from']
                l_val = pyo.value(model.l[k, t])
                v_val = pyo.value(model.v[fb, t])
                P_val = pyo.value(model.P[k, t])
                Q_val = pyo.value(model.Q[k, t])
                
                gap = abs((l_val * v_val) - (P_val**2 + Q_val**2))
                if gap > max_gap:
                    max_gap = gap
        return max_gap
    except Exception:
        return -1.0

def calculate_performance_index(model):
    perf = 0.0
    for i in model.N:
        for t in model.T:
            perf += pyo.value(model.v_viol_down[i, t]) + pyo.value(model.v_viol_up[i, t])
    return max(0.0, perf)

def self_play(start_ep=0, total_eps=200, resume_model=None):
    env = ActivePlanningEnv()
    net = ADNDeepNet(num_buses=33, num_device_types=4)
    optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-4) 
    
    buffer = ReplayBuffer(capacity=10000) 
    checkpoint_interval = 50 
    
    INITIAL_TEMP = 1.0
    MIN_TEMP = 0.1
    DECAY_RATE = 0.98 
    
    training_history = []
    
    print("==================================================")
    print("   Starting AlphaZero Self-Play Training Loop")
    print("   Physics Engine: Convex SOCP Relaxation (DistFlow)")
    print(f"  Target Episodes: {start_ep + 1} to {total_eps}")
    print("==================================================\n")

    if resume_model and os.path.exists(resume_model):
        net.load_state_dict(torch.load(resume_model, weights_only=True))
        print(f" [*] SUCCESSFULLY LOADED WEIGHTS FROM: {resume_model}\n")
    else:
        print(f" [*] Starting fresh from scratch. No previous weights loaded.\n")

    print("  Evaluating Base Network for Reward Scaling...")
    empty_placement = {'ess': [], 'pv': [], 'gas': [], 'svc': [], 'cb': []}
    is_base_feasible, base_cost = env.base_model_evaluate(empty_placement)
    base_violations = calculate_performance_index(env.base_model)
    print(f"  --> BASE NETWORK VIOLATIONS (NO AI): {base_violations:.4f}")

    min_v = 1.5
    for i in env.base_model.N:
        for t in env.base_model.T:
            v_val = math.sqrt(pyo.value(env.base_model.v[i, t]))
            if v_val < min_v:
                min_v = v_val
    print(f"  --> BASE NETWORK MIN VOLTAGE: {min_v:.4f} p.u.")
    print("--------------------------------------------------")
    
    if not is_base_feasible or base_cost > 10000000:
        base_cost = 46000000.0
    print(f"   Base Cost established at: ${base_cost:,.2f}")
    print("--------------------------------------------------")

    for ep in range(start_ep, total_eps):
        print(f"\n--- Episode {ep+1}/{total_eps} ---")
        state = env.reset()
        
        current_temp = max(MIN_TEMP, INITIAL_TEMP * (DECAY_RATE ** (ep - start_ep)))
        print(f"  Current MCTS Temperature: {current_temp:.3f}")
        
        mcts = MCTS(neural_net=net, num_simulations=400)
        episode_memory = []
        step = 0
        final_cost = 0.0 
        actual_perf = 100.0
        gap = 1.0 
        
        while True:
            best_action, action_probs = mcts.search(state, temperature=current_temp, add_noise=True)
            
            if best_action is None:
                print("       No valid actions left. Ending episode.")
                break
                
            state_tensor = mcts.state_to_tensor(state)
            episode_memory.append([state_tensor, action_probs, best_action, 0.0])

            if best_action[0] == 'stop':
                print(f"       [Step {step+1}] AI chosen action -> STOP INVESTING")
            else:
                print(f"       [Step {step+1}] AI chosen action -> Device: {best_action[0].upper()}, Bus: {best_action[1]}")

            state, reward, is_infeasible_done, info = env.step(best_action)
            msg = info.get("msg", "")

            if is_infeasible_done:
                if "Feasible" in msg:
                    gap = verify_exact_physics(env.base_model)
                    actual_perf = calculate_performance_index(env.base_model)
                    episode_memory[-1][3] = actual_perf
                    final_cost = info.get('cost', base_cost)
                    print(f"       Optimal Design Reached! Cost: ${final_cost:,.0f} | Gap: {gap:.2e} | Violations: {actual_perf:.4f}")
                else:
                    print(f"       Blackout! Physical limits exceeded. Solver hit '{msg}'.")
                    episode_memory[-1][3] = 100.0
                    final_cost = base_cost * (1.5 - (step * 0.05))
                break 

            step += 1
            
        # Piecewise
        LAMBDA_V = 100000.0      
        LAMBDA_GAP = 5000000.0   
        
        MIN_FEASIBLE_COST = 100000.0   
        MAX_FEASIBLE_COST = 3000000.0  

        for seq in episode_memory:
            s_tensor, t_policy, _, target_perf = seq
            
            effective_perf = 0.0 if target_perf < 1e-4 else target_perf
            effective_gap = 0.0 if gap < 1e-3 else gap  
            
            if effective_perf > 0.0 or effective_gap > 0.0:
                penalty = (LAMBDA_V * effective_perf) + (LAMBDA_GAP * effective_gap)
                scaled_value = max(-1.0, - (penalty / 500000.0))
            else:
                scaled_value = 1.0 - ((final_cost - MIN_FEASIBLE_COST) / (MAX_FEASIBLE_COST - MIN_FEASIBLE_COST))
                scaled_value = max(0.0, min(1.0, scaled_value))
                
            buffer.push(s_tensor, t_policy, scaled_value, effective_perf)
        
        ep_tot_loss, ep_p_loss, ep_v_loss = 0.0, 0.0, 0.0
        
        
        if len(buffer) > 256:
            epochs_per_episode = 3 
            tot_losses, p_losses, v_losses = [], [], []
            for _ in range(epochs_per_episode):
                states, target_pis, target_values, target_perfs = buffer.sample(256)
                tl, pl, vl = net.train_step(optimizer, states, target_pis, target_values, target_perfs)
                tot_losses.append(tl)
                p_losses.append(pl)
                v_losses.append(vl)
                
            ep_tot_loss = sum(tot_losses) / len(tot_losses)
            ep_p_loss = sum(p_losses) / len(p_losses)
            ep_v_loss = sum(v_losses) / len(v_losses)
            print(f"    [NN Update] Average Loss: {ep_tot_loss:.4f} (Policy: {ep_p_loss:.4f}, Value: {ep_v_loss:.4f})")

        training_history.append({
            'Episode': ep + 1,
            'Temperature': round(current_temp, 3),
            'Final_Cost($)': round(final_cost, 2),
            'Violations(p.u.)': round(actual_perf, 4),
            'Total_Loss': round(ep_tot_loss, 4),
            'Policy_Loss': round(ep_p_loss, 4),
            'Value_Loss': round(ep_v_loss, 4)
        })

        if (ep + 1) % checkpoint_interval == 0:
            checkpoint_name = f"trained_adn_net_checkpoint_ep{ep+1}.pth"
            torch.save(net.state_dict(), checkpoint_name)
            print(f"    [CHECKPOINT] Model saved safely to {checkpoint_name}")

    print("\n  Training Complete! Saving final models and logs...")
    final_model_name = f"trained_adn_net_ep{total_eps}.pth"
    torch.save(net.state_dict(), final_model_name)
    print(f" Final weights saved as: {final_model_name}")
    
    df = pd.DataFrame(training_history)
    csv_filename = f"training_log_{start_ep+1}_to_{total_eps}.csv"
    df.to_csv(csv_filename, index=False)
    print(f" [!] Learning Curve Data successfully exported to: {csv_filename}")

if __name__ == "__main__":
    self_play(start_ep=0, total_eps=200, resume_model=None)