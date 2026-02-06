"""
=============================================================================
AC AGENT - Actor-Critic Algorithm Implementation
=============================================================================
Implements the classic Actor-Critic algorithm for continuous control.

KEY DIFFERENCES FROM DDPG:
    - Stochastic policy (Actor outputs Normal dist, samples action)
    - Critic outputs V(s) (state value), not Q(s,a)
    - Uses TD error as advantage: TD_error = r + gamma*V(s') - V(s)
    - Actor loss: -log_prob(a) * TD_error (advantage-weighted policy gradient)

ALGORITHM:
    Critic: MSE loss between V(s) and target_V = r + gamma * V_target(s')
    Actor:  Minimize -log_prob(a) * TD_error (increase log_prob of good actions)
=============================================================================
"""

import copy

import numpy as np
import torch
import torch.nn as nn

import utils.ounoise
import utils.dataloger
from ac.actor_critic import ActorCritic, Actor, Critic


class ActorCriticAgent:
    """
    Actor-Critic agent: stochastic policy + state-value critic.
    Uses replay buffer and target critic (like DDPG) for stability.
    """

    def __init__(self, env, hidden1, hidden2, gamma, lr_actor=0.00001, lr_critic=0.0001,
                 buf_size=1000000, sync_freq=100, batch_size=64, exp_name='exp1', device='cuda'):
        """
        Initialize AC agent: Actor, Critic, target Critic, replay buffer, OU noise.
        """
        self.env = env
        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.shape[0]
        self.device = device

        self.actor = Actor(self.n_state, self.n_action, hidden1, hidden2).to(device)
        self.critic = Critic(self.n_state, self.n_action, hidden1, hidden2).to(device)
        self.target_critic = copy.deepcopy(self.critic)  # For stable V(s') targets

        # Experience replay (same structure as DDPG)
        self.buf_size = buf_size
        self.buffer = np.zeros((self.buf_size, self.n_state * 2 + 3))
        self.bf_counter = 0
        self.learn_counter = 0
        self.sync_freq = sync_freq

        # OU noise for additional exploration (MountainCar is difficult)
        self.explore_mu = 0.2
        self.explore_theta = 0.15
        self.explore_sigma = 0.2
        self.noise = utils.ounoise.OUNoise(
            self.n_action, self.explore_mu, self.explore_theta, self.explore_sigma
        )

        # Hyperparameters and optimizers
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.exp_name = exp_name
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.mse_loss = nn.MSELoss()
        self.loger = utils.dataloger.DataLoger('./result/ac/' + self.exp_name)

    def save_transition(self, state, action, reward, nxt_state, done):
        """Store transition (s, a, r, s', done) in replay buffer."""
        idx = self.bf_counter % self.buf_size
        self.buffer[idx, :] = np.hstack((state, action, reward, nxt_state, done))
        self.bf_counter += 1

    def predict(self, state):
        """
        Sample action from actor's policy (no extra noise).
        For AC: samples from Normal(mu, sigma).
        """
        state = torch.Tensor(state).to(self.device)
        dist = self.actor(state)
        action = dist.sample().cpu().numpy()
        return action

    def sample_act(self, state, noise_scale=1.0):
        """
        Sample action for training: actor dist.sample() + OU noise.
        """
        state = torch.Tensor(state).to(self.device)
        dist = self.actor(state)
        action = dist.sample().cpu().numpy()
        return action + self.noise.sample() * noise_scale

    def sample_batch(self):
        """Randomly sample batch from replay buffer."""
        max_buffer = min(self.bf_counter, self.buf_size)
        idx = np.random.choice(max_buffer, self.batch_size, replace=False)
        batch = self.buffer[idx, :]
        state = batch[:, :self.n_state]
        action = batch[:, self.n_state:self.n_state + 1]
        reward = batch[:, self.n_state + 1:self.n_state + 2]
        nxt_state = batch[:, self.n_state + 2:self.n_state * 2 + 2]
        done = batch[:, self.n_state * 2 + 2:]
        return state, action, reward, nxt_state, done

    def learn(self, epoch, step, cur_state):
        """
        One AC learning step.

        TD_ERROR = r + gamma * V(s') - V(s)  (advantage estimate)
        Actor:  minimize -log_prob(a) * TD_error
               (increase probability of actions with positive TD_error)
        Critic: minimize MSE(V(s), r + gamma * V_target(s'))
        """
        max_buffer = min(self.bf_counter, self.buf_size)
        if max_buffer < self.batch_size:
            return

        self.learn_counter += 1
        if self.learn_counter % self.sync_freq == 0:
            self.target_critic.load_state_dict(self.critic.state_dict())

        state, action, reward, nxt_state, done = self.sample_batch()
        state = torch.Tensor(state).to(self.device)
        action = torch.Tensor(action).to(self.device)
        reward = torch.Tensor(reward).to(self.device)
        nxt_state = torch.Tensor(nxt_state).to(self.device)
        done = torch.Tensor(done).to(self.device)

        # ---------------------------------------------------------------------
        # Compute TD error (advantage estimate)
        # ---------------------------------------------------------------------
        # TD_error = r + gamma*V(s') - V(s)  (target - current estimate)
        # Positive TD_error = action was better than expected
        dist, V = self.actor(state), self.critic(state)
        nxt_V = self.target_critic(nxt_state).detach()
        TD_error = reward + self.gamma * nxt_V * (1 - done) - V

        # ---------------------------------------------------------------------
        # Actor update: advantage-weighted policy gradient
        # ---------------------------------------------------------------------
        # Loss = -log_prob(a) * TD_error
        # TD_error.detach() prevents gradients flowing to critic
        # Large TD_error -> larger actor update (action was surprisingly good)
        actor_loss = -dist.log_prob(action) * TD_error.detach()
        actor_loss = actor_loss.mean()

        # ---------------------------------------------------------------------
        # Critic update: fit V(s) to Bellman target
        # ---------------------------------------------------------------------
        critic_loss = self.mse_loss(V, reward + self.gamma * nxt_V * (1 - done))

        # Update actor (with gradient clipping for stability)
        self.optim_actor.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 3)
        self.optim_actor.step()

        # Update critic
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        if step % 100 == 0:
            self.loger.log('actor_loss', actor_loss.item(), step)
            self.loger.log('critic_loss', critic_loss.item(), step)
            if step % 1000 == 0:
                print('epoch: {}, step: {}, actor_loss: {}, critic_loss: {}, TD_error:{}'.format(
                    epoch, step, actor_loss, critic_loss, TD_error.mean()))

    def save_model(self, epoch, avg_score):
        """Save actor and critic to result/ac/{exp_name}/"""
        torch.save(self.actor.state_dict(),
                   './result/ac/' + self.exp_name +
                   '/actor_epoch{}_avgScore{:.3f}.pth'.format(epoch, avg_score))
        torch.save(self.critic.state_dict(),
                   './result/ac/' + self.exp_name +
                   '/critic_epoch{}_avgScore{:.3f}.pth'.format(epoch, avg_score))

    def load_model(self, actor_path, critic_path):
        """Load actor and critic weights."""
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
