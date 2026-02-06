"""
=============================================================================
DDPG AGENT - Deep Deterministic Policy Gradient Implementation
=============================================================================
Implements the DDPG algorithm (Continuous control with deep RL, Lillicrap et al.).

KEY COMPONENTS:
    - Actor/Critic networks + Target networks (for stable Q-learning)
    - Experience replay buffer (off-policy learning)
    - Ornstein-Uhlenbeck (OU) noise for exploration
    - Bellman update for critic, policy gradient for actor

ALGORITHM:
    Critic: MSE loss between Q(s,a) and target_Q = r + gamma * Q_target(s', actor_target(s'))
    Actor:  Maximize Q(s, actor(s)) -> loss = -Q(s, actor(s))
=============================================================================
"""

import copy

import numpy as np
import torch

import utils.ounoise
import utils.dataloger
from ddpg.actor_critic import DDPGActor, DDPGCritic


class DDPGAgent:
    """
    DDPG Agent: manages actor, critic, replay buffer, exploration noise,
    and the learning process.
    """

    def __init__(self, env, hidden1, hidden2, gamma, lr_actor=0.001, lr_critic=0.001,
                 buf_size=1000000, sync_freq=100, batch_size=64, exp_name='exp1', device='cuda'):
        """
        Initialize DDPG agent with networks, target networks, replay buffer, OU noise.
        """
        self.env = env
        self.n_state = env.observation_space.shape[0]
        self.n_action = env.action_space.shape[0]
        self.device = device

        # ---------------------------------------------------------------------
        # Actor and Critic networks
        # ---------------------------------------------------------------------
        self.actor = DDPGActor(self.n_state, self.n_action, hidden1, hidden2).to(device)
        self.critic = DDPGCritic(self.n_state, self.n_action, hidden1, hidden2).to(device)

        # Target networks: slow-moving copies for stable Q-learning targets
        # Updated every sync_freq steps (hard update)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)

        # ---------------------------------------------------------------------
        # Ornstein-Uhlenbeck noise for exploration
        # ---------------------------------------------------------------------
        # MountainCar is difficult; deterministic policy without noise rarely
        # discovers the solution (swinging back and forth). OU noise adds
        # correlated exploration to the action.
        self.explore_mu = 0.2
        self.explore_theta = 0.15
        self.explore_sigma = 0.2
        self.noise = utils.ounoise.OUNoise(
            self.n_action, self.explore_mu, self.explore_theta, self.explore_sigma
        )

        # ---------------------------------------------------------------------
        # Experience replay buffer
        # ---------------------------------------------------------------------
        # Stores transitions (s, a, r, s', done). Each row: n_state + 1 + 1 + n_state + 1 = n_state*2 + 3
        self.buf_size = buf_size
        self.buffer = np.zeros((self.buf_size, self.n_state * 2 + 3))
        self.bf_counter = 0   # Total transitions stored
        self.learn_counter = 0
        self.sync_freq = sync_freq  # Steps between target network updates

        # ---------------------------------------------------------------------
        # Hyperparameters and optimizers
        # ---------------------------------------------------------------------
        self.gamma = gamma
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.batch_size = batch_size
        self.exp_name = exp_name
        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.mse_loss = torch.nn.MSELoss()
        self.loger = utils.dataloger.DataLoger('./result/ddpg/' + self.exp_name)

    def learn(self, epoch, step, cur_state):
        """
        One DDPG learning step: update critic and actor using a batch from replay buffer.

        CRITIC: Minimize MSE between Q(s,a) and Bellman target:
            target_Q = r + gamma * Q_target(s', actor_target(s')) * (1 - done)

        ACTOR: Maximize Q(s, actor(s)) -> minimize -Q(s, actor(s))
        """
        max_buffer = min(self.bf_counter, self.buf_size)
        if max_buffer < self.batch_size:
            return

        self.learn_counter += 1

        # Hard update target networks every sync_freq steps
        if self.learn_counter % self.sync_freq == 0:
            self.target_actor.load_state_dict(self.actor.state_dict())
            self.target_critic.load_state_dict(self.critic.state_dict())

        # Sample batch of transitions from replay buffer
        state, action, reward, nxt_state, done = self.sample_batch()
        state = torch.Tensor(state).to(self.device)
        action = torch.Tensor(action).to(self.device)
        reward = torch.Tensor(reward).to(self.device)
        nxt_state = torch.Tensor(nxt_state).to(self.device)
        done = torch.Tensor(done).to(self.device)

        # ---------------------------------------------------------------------
        # Critic update: Bellman equation
        # ---------------------------------------------------------------------
        # target_Q = r + gamma * Q_target(s', a')  where a' = actor_target(s')
        # (1 - done) zeros future Q when episode ends
        nxt_action = self.target_actor(nxt_state)
        nxt_q = self.target_critic(nxt_state, nxt_action)
        target_q = reward + self.gamma * nxt_q * (1 - done)
        target_q = target_q.detach()  # No gradients through target

        q = self.critic(state, action)
        critic_loss = self.mse_loss(q, target_q)

        # ---------------------------------------------------------------------
        # Actor update: policy gradient
        # ---------------------------------------------------------------------
        # Maximize Q(s, actor(s)) -> minimize -Q(s, actor(s))
        # Actor output is fed to critic; gradient flows through critic to actor
        # (Uses target_critic for stability; main critic also valid)
        actor_loss = -self.target_critic(state, self.actor(state)).mean()

        # Update critic
        self.optim_critic.zero_grad()
        critic_loss.backward()
        self.optim_critic.step()

        # Update actor
        self.optim_actor.zero_grad()
        actor_loss.backward()
        self.optim_actor.step()

        # Log to TensorBoard
        if step % 100 == 0:
            self.loger.log('actor_loss', actor_loss.item(), step)
            self.loger.log('critic_loss', critic_loss.item(), step)
            if step % 1000 == 0:
                print('epoch: {}, step: {}, actor_loss: {}, critic_loss: {}, '
                      'cur_state:[{},{}]'.format(
                          epoch, step, actor_loss, critic_loss, cur_state[0], cur_state[1]))

    def predict(self, state):
        """
        Return greedy action (no noise). Used during evaluation.
        :param state: (n_state,)
        :return: action (n_action,)
        """
        state = torch.Tensor(state).to(self.device).view(1, -1)
        action = self.actor(state).cpu().detach().numpy()[0]
        return action

    def save_transition(self, state, action, reward, nxt_state, done):
        """
        Store one transition (s, a, r, s', done) in the replay buffer.
        Buffer layout per row: [state | action | reward | nxt_state | done]
        """
        idx = self.bf_counter % self.buf_size  # Circular buffer
        self.buffer[idx, :] = np.hstack((state, action, reward, nxt_state, done))
        self.bf_counter += 1

    def sample_act(self, state, noise_scale=1.0):
        """
        Sample action for training: actor output + OU noise for exploration.
        :param state: (n_state,)
        :param noise_scale: scale factor for noise
        :return: action (n_action,) with exploration noise
        """
        state = torch.Tensor(state).to(self.device).view(1, -1)
        action = self.actor(state).cpu().detach().numpy()[0]
        return action + self.noise.sample() * noise_scale

    def sample_batch(self):
        """
        Randomly sample batch_size transitions from the replay buffer.
        :return: (state, action, reward, nxt_state, done) each as ndarray
        """
        max_buffer = min(self.bf_counter, self.buf_size)
        idx = np.random.choice(max_buffer, self.batch_size, replace=False)
        batch = self.buffer[idx, :]
        state = batch[:, :self.n_state]
        action = batch[:, self.n_state:self.n_state + 1]
        reward = batch[:, self.n_state + 1:self.n_state + 2]
        nxt_state = batch[:, self.n_state + 2:self.n_state * 2 + 2]
        done = batch[:, self.n_state * 2 + 2:]
        return state, action, reward, nxt_state, done

    def save_model(self, epoch, avg_score):
        """Save actor and critic weights to result/ddpg/{exp_name}/"""
        torch.save(self.actor.state_dict(),
                   './result/ddpg/' + self.exp_name +
                   '/actor_epoch{}_avgScore{:.3f}.pth'.format(epoch, avg_score))
        torch.save(self.critic.state_dict(),
                   './result/ddpg/' + self.exp_name +
                   '/critic_epoch{}_avgScore{:.3f}.pth'.format(epoch, avg_score))

    def load_model(self, actor_path, critic_path):
        """Load actor and critic weights from disk."""
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
