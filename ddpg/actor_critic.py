"""
=============================================================================
DDPG ACTOR-CRITIC NETWORKS
=============================================================================
Contains the two neural networks used by the DDPG algorithm:

1. DDPGActor:   Deterministic policy - maps state -> action
2. DDPGCritic: Q-value function - maps (state, action) -> Q(s,a)

DDPG uses a DETERMINISTIC policy (unlike AC which uses stochastic).
For MountainCarContinuous: state dim=2, action dim=1.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# DDPGActor: Deterministic Policy Network
# -----------------------------------------------------------------------------
# Input: state (batch_size, n_state)
# Output: action (batch_size, n_action), bounded in [-1, 1] via Tanh
#
# The actor learns to output actions that MAXIMIZE the critic's Q(s,a).
# Tanh on output ensures actions stay in valid range for MountainCar.
class DDPGActor(nn.Module):
    def __init__(self, n_state, n_action, hidden1, hidden2):
        super().__init__()
        n_action = 1  # MountainCarContinuous has 1 continuous action (force)
        # MLP: state -> hidden1 -> hidden2 -> action
        self.fnn = nn.Sequential(
            nn.Linear(n_state, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, n_action),
            nn.Tanh()  # Bounds action to [-1, 1] (left/right force)
        )

    def forward(self, inputs):
        """
        Forward pass: state -> deterministic action.
        :param inputs: state, shape [batch_size, n_state]
        :return: action, shape [batch_size, n_action] in [-1, 1]
        """
        return self.fnn(inputs)


# -----------------------------------------------------------------------------
# DDPGCritic: Q-Value Network
# -----------------------------------------------------------------------------
# Input: state (batch_size, n_state), action (batch_size, n_action)
# Output: Q(s,a) scalar (batch_size, 1) - expected cumulative future reward
#
# Uses two parallel streams (state and action) that are combined before output.
# This architecture allows efficient computation of Q(s, actor(s)) for gradients.
class DDPGCritic(nn.Module):
    def __init__(self, n_state, n_action, hidden1, hidden2):
        super().__init__()
        # State stream: state -> hidden1 -> hidden2
        self.fnn_state = nn.Sequential(
            nn.Linear(n_state, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2)
        )
        # Action stream: action -> hidden1 -> hidden2
        self.fnn_action = nn.Sequential(
            nn.Linear(n_action, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2)
        )
        # Combine streams: element-wise add, then ReLU, then output Q
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden2, 1)

    def forward(self, state, action):
        """
        Forward pass: (state, action) -> Q(s,a).
        :param state: shape [batch_size, n_state]
        :param action: shape [batch_size, n_action]
        :return: Q(s,a), shape [batch_size, 1]
        """
        x_state = self.fnn_state(state)
        x_action = self.fnn_action(action)
        x = x_state + x_action  # Combine state and action representations
        x = self.relu(x)
        return self.output_layer(x)
