# core / network
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ADNDeepNet(nn.Module):
    def __init__(self, num_buses=33, num_device_types=4):
        super(ADNDeepNet, self).__init__()
        self.num_buses = num_buses
        self.num_device_types = num_device_types
        self.input_dim = num_buses * num_device_types
        self.policy_dim = self.input_dim + 1 

        # 1. Policy Network
        self.pol_fc1 = nn.Linear(self.input_dim, 256)
        self.pol_fc2 = nn.Linear(256, 128)
        self.pol_out = nn.Linear(128, self.policy_dim) # اصلاح شد

        # 2. Value Network
        self.val_fc1 = nn.Linear(self.input_dim, 256)
        self.val_fc2 = nn.Linear(256, 64)
        self.val_out = nn.Linear(64, 1)

        # 3. Performance Network
        self.prf_fc1 = nn.Linear(self.input_dim, 128)
        self.prf_fc2 = nn.Linear(128, 64)
        self.prf_out = nn.Linear(64, 1)

    def forward(self, x):
        p = F.relu(self.pol_fc1(x))
        p = F.relu(self.pol_fc2(p))
        logits = self.pol_out(p)
        
        
        if logits.dim() == 1:
            logits[0:4] = -1e9
        else:
            logits[:, 0:4] = -1e9
            
        log_policy_probs = F.log_softmax(logits, dim=-1)

        v = F.relu(self.val_fc1(x))
        v = F.relu(self.val_fc2(v))
        value = torch.tanh(self.val_out(v))
        
        pr = F.relu(self.prf_fc1(x))
        pr = F.relu(self.prf_fc2(pr))
        perf_index = F.relu(self.prf_out(pr))

        return log_policy_probs, value, perf_index


    def predict(self, state_tensor):
        self.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state_tensor)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
                
            log_policy_probs, value, perf_index = self.forward(state_tensor)
            
            policy_probs = torch.exp(log_policy_probs)
            
        return policy_probs.squeeze(0).numpy(), value.item(), perf_index.item()

    def train_step(self, optimizer, states, target_pis, target_values, target_perfs=None):
        self.train()
        optimizer.zero_grad()

        states = torch.FloatTensor(np.array(states))
        target_pis = torch.FloatTensor(np.array(target_pis))
        target_values = torch.FloatTensor(np.array(target_values)).unsqueeze(1)
        
        if target_perfs is None:
            target_perfs = torch.zeros_like(target_values)
        else:
            target_perfs = torch.FloatTensor(np.array(target_perfs)).unsqueeze(1)

        log_policy_probs, values, perfs = self.forward(states)

        loss_policy = F.kl_div(log_policy_probs, target_pis, reduction='batchmean')
        loss_value = F.mse_loss(values, target_values)
        loss_perf = F.mse_loss(perfs, target_perfs)
        
        total_loss = loss_policy + 0.5 * loss_value + 0.001 * loss_perf
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        optimizer.step()

        return total_loss.item(), loss_policy.item(), loss_value.item()   
