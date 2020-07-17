import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
import numpy as np

class MemoryBuffer:
    '''Simple buffer to collect experiences and clear after each update.'''
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.state_values = []
    
    def clear_buffer(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.state_values[:]

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, action_std=0.5, hidden_size=32, low_policy_weights_init=True):
        super().__init__()

        self.actor_fc1 = nn.Linear(state_size, 2*hidden_size)
        self.actor_fc2 = nn.Linear(2*hidden_size, hidden_size)

        self.actor_mu = nn.Linear(hidden_size, action_size)
        self.actor_sigma = nn.Linear(hidden_size, action_size)
        
        
        self.critic_fc1 = nn.Linear(state_size, 2*hidden_size)
        self.critic_fc2 = nn.Linear(2*hidden_size, hidden_size)

        self.critic_value = nn.Linear(hidden_size, 1)

        self.distribution = torch.distributions.Normal

        self.action_var = torch.full((action_size,), action_std*action_std)
        
        # Boosts training performance in the beginning
        if low_policy_weights_init:
            with torch.no_grad():
                self.actor_mu.weight.mul_(0.01)

    def forward(self, state):
        x = torch.tanh(self.actor_fc1(state))
        x = torch.tanh(self.actor_fc2(x))
        mu = torch.tanh(self.actor_mu(x))
        sigma = F.softplus(self.actor_sigma(x))

        v = torch.tanh(self.critic_fc1(state))
        v = torch.tanh(self.critic_fc2(v))
        state_value = self.critic_value(v)

        return mu, sigma, state_value 

    def act(self, state, memory):
        '''Choose action according to the policy.'''
        action_mu, action_sigma, state_value = self.forward(state)

        action_var = self.action_var.expand_as(action_mu)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mu, cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(log_prob)

        return action.detach()
    

    def evaluateStd(self, state, action):
        '''Evaluate action using learned std value for distribution.'''
        action_mu, action_sigma, state_value = self.forward(state)
        m = self.distribution(action_mu.squeeze(), action_sigma.squeeze())
        log_prob = m.log_prob(action)

        return log_prob, state_value

    def evaluate(self, state, action):
        '''Evaluate action for a given state.'''   
        action_mean, _, state_value = self.forward(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO():
    '''Proximal Policy Optimization algorithm.'''
    def __init__(self, state_size, action_size, lr=1e-4, gamma=0.99, epsilon_clip=0.2, epochs=80, action_std=0.5):

        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.gamma  = gamma
        self.epsilon_clip = epsilon_clip
        self.K_epochs = epochs

        self.policy = ActorCritic(self.state_size, self.action_size, action_std)
        self.policy_old = ActorCritic(self.state_size, self.action_size, action_std)

        self.MseLoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=(0.9, 0.999))
    
    def select_action(self, state, memory):
        '''Get action using state in numpy format'''
        state = torch.FloatTensor(state.reshape(1, -1))
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()

    def update(self, memory):
        '''Update agent's network using collected set of experiences.'''
        states = memory.states
        actions = memory.actions
        rewards = memory.rewards
        dones = memory.dones
        log_probs = memory.logprobs 

        discounted_rewards = []
        discounted_reward = 0
        for i in reversed(range(len(rewards))):
            if dones[i] == True:
                discounted_reward = 0  
            discounted_reward = rewards[i] + self.gamma*discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        # old_state_values = torch.stack(state_values, 1).detach()
        # advantages = discounted_rewards - old_state_values.detach().squeeze()
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        
        states = torch.squeeze(torch.stack(states), 1).detach()
        actions = torch.squeeze(torch.stack(actions), 1).detach()
        old_log_probs = torch.squeeze(torch.stack(log_probs), 1).detach()

        for epoch in range(self.K_epochs):

            new_log_probs, state_values, dist_entropy = self.policy.evaluate(states, actions)

            new_log_probs = new_log_probs.squeeze()
            advantages = discounted_rewards - state_values.detach().squeeze()
            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            ratios_clipped = torch.clamp(ratios, min=1-self.epsilon_clip, max=1+self.epsilon_clip)
            loss = -torch.min(ratios*advantages, ratios_clipped*advantages)+ 0.5*self.MseLoss(state_values, discounted_rewards) - 0.01*dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
