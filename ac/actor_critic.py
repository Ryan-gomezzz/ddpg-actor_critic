"""
=============================================================================
AC ACTOR-CRITIC NETWORKS
=============================================================================
Contains the neural networks for the Actor-Critic (AC) algorithm.

KEY DIFFERENCES FROM DDPG:
    1. Actor: STOCHASTIC policy - outputs Normal(mu, sigma), samples action
    2. Critic: V(s) (state value) - NOT Q(s,a); critic takes only state
    3. AC uses TD error as advantage; DDPG uses Q-learning

CLASSES:
    Actor:   state -> Normal(mu, sigma) distribution
    Critic:  state -> V(s) scalar
    ActorCritic: combined shared-backbone version (not used by ac_agent)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Actor: Stochastic Policy Network
# -----------------------------------------------------------------------------
# Outputs a probability distribution over actions (Normal), not a single action.
# Actions are SAMPLED from the distribution for exploration.
# mu: mean action (bounded by 2*tanh -> [-2, 2] for MountainCar)
# sigma: std (via softplus to keep positive)
class Actor(nn.Module):
    def __init__(self, n_state, n_action, hidden1, hidden2):
        super().__init__()
        self.fc1 = nn.Linear(n_state, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.mu_layer = nn.Linear(hidden2, n_action)
        self.sigma_layer = nn.Linear(hidden2, n_action)
        self.distribution = torch.distributions.Normal

    def forward(self, state):
        """
        Forward: state -> Normal(mu, sigma) distribution.
        :param state: (batch_size, n_state)
        :return: dist (torch.distributions.Normal) - call dist.sample() for action
        """
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        mu = 2 * torch.tanh(self.mu_layer(x))  # Mean in [-2, 2] for action range
        sigma = F.softplus(self.sigma_layer(x)) + 1e-5  # Std > 0, avoid 0
        dist = self.distribution(mu, sigma)
        return dist


# -----------------------------------------------------------------------------
# Critic: State Value Network
# -----------------------------------------------------------------------------
# Outputs V(s) - expected return from state s - NOT Q(s,a).
# Unlike DDPG critic which takes (state, action) and outputs Q(s,a).
class Critic(nn.Module):
    def __init__(self, n_state, n_action, hidden1, hidden2):
        super().__init__()
        self.fc1 = nn.Linear(n_state, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.V_layer = nn.Linear(hidden2, n_action)

    def forward(self, state):
        """
        Forward: state -> V(s) (state value).
        :param state: (batch_size, n_state)
        :return: V, shape (batch_size, n_action) - expected return from state
        """
        x = self.fc1(state)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        V = self.V_layer(x)
        return V


# -----------------------------------------------------------------------------
# ActorCritic: Combined Module (shared backbone)
# -----------------------------------------------------------------------------
# Single module with shared fc layers; outputs both policy dist and V(s).
# NOTE: ac_agent.py uses separate Actor and Critic; this class is defined
# but not currently used in the training loop.
class ActorCritic(nn.Module):
    def __init__(self, n_state, n_action, hidden1, hidden2):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(n_state, hidden1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.mu_layer = nn.Linear(hidden2, n_action)
        self.sigma_layer = nn.Linear(hidden2, n_action)
        self.V_layer = nn.Linear(hidden2, n_action)
        self.distribution = torch.distributions.Normal

    def forward(self, x):
        """
        Shared backbone; outputs (dist, V).
        :param x: state batch (batch_size, n_state)
        :return: (dist, V)
        """
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        mu = 2 * torch.tanh(self.mu_layer(x))
        sigma = F.softplus(self.sigma_layer(x)) + 1e-5
        dist = self.distribution(mu.view(1, -1).data, sigma.view(1, -1).data)
        V = self.V_layer(x)
        return dist, V
