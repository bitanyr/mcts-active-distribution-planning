"""Policy, value, and two-component performance network."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data.devices import DEVICE_TYPES
from data.ieee33 import NUM_BUSES


class ADNDeepNet(nn.Module):
    def __init__(self, num_buses=NUM_BUSES, num_device_types=len(DEVICE_TYPES)):
        super().__init__()
        self.num_buses = int(num_buses)
        self.num_device_types = int(num_device_types)
        # The substation bus is not an installation candidate.
        self.input_dim = (self.num_buses - 1) * self.num_device_types
        self.policy_dim = self.input_dim + 1

        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(128, self.policy_dim)
        self.value_head = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.performance_head = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, x):
        features = self.trunk(x)
        log_policy = F.log_softmax(self.policy_head(features), dim=-1)
        value = torch.tanh(self.value_head(features))
        performance = torch.sigmoid(self.performance_head(features))
        return log_policy, value, performance

    def predict(self, state_tensor):
        self.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(state_tensor, dtype=torch.float32)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            log_policy, value, performance = self.forward(tensor)
        return (
            log_policy.exp().squeeze(0).cpu().numpy(),
            float(value.item()),
            performance.squeeze(0).cpu().numpy(),
        )

    def train_step(self, optimizer, states, target_policies, target_values, target_performances):
        self.train()
        optimizer.zero_grad()
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        target_policies = torch.as_tensor(np.asarray(target_policies), dtype=torch.float32)
        target_values = torch.as_tensor(np.asarray(target_values), dtype=torch.float32).reshape(-1, 1)
        target_performances = torch.as_tensor(
            np.asarray(target_performances), dtype=torch.float32
        ).reshape(-1, 2)

        log_policy, values, performances = self.forward(states)
        policy_loss = -(target_policies * log_policy).sum(dim=1).mean()
        value_loss = F.mse_loss(values, target_values)
        performance_loss = F.mse_loss(performances, target_performances)
        total_loss = policy_loss + 0.5 * value_loss + 0.1 * performance_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return {
            "total": float(total_loss.item()),
            "policy": float(policy_loss.item()),
            "value": float(value_loss.item()),
            "performance": float(performance_loss.item()),
        }