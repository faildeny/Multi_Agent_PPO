import torch
from torch import nn
import torch.nn.functional as F
import gym
from torch.distributions import Categorical
import numpy as np
from collections import deque

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super().__init__()

        self.actor_fc1 = nn.Linear(state_size, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)

        self.actor_mu = nn.Linear(hidden_size, action_size)
        self.actor_sigma = nn.Linear(hidden_size, action_size)
        
        
        self.critic_fc1 = nn.Linear(state_size, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)

        self.critic_value = nn.Linear(hidden_size, 1)

        self.distribution = torch.distributions.Normal

    def forward(self, state):
        x = torch.tanh(self.actor_fc1(state))
        x = torch.tanh(self.actor_fc2(x))
        mu = torch.tanh(self.actor_mu(x))
        sigma = F.softplus(self.actor_sigma(x))

        v = torch.tanh(self.critic_fc1(state))
        v = torch.tanh(self.critic_fc2(v))
        state_value = self.critic_value(v)

        return mu, sigma, state_value 

    def act(self, state):
        action_mu, action_sigma, _ = self.forward(state)
        m = self.distribution(action_mu, action_sigma)
        action = m.sample()
        log_prob = m.log_prob(action)

        return action, log_prob
    
    def evaluate(self, state, action):
        action_mu, action_sigma, state_value = self.forward(state)
        m = self.distribution(action_mu.squeeze(), action_sigma.squeeze())
        log_prob = m.log_prob(action)

        return log_prob, state_value


class PPO():
    def __init__(self, env):

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.lr = 0.0003
        self.gamma  = 0.99
        self.epsilon = 0.2
        self.K_epochs = 80

        self.policy = ActorCritic(self.state_size, self.action_size)
        self.policy_old = ActorCritic(self.state_size, self.action_size)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def update(self, states, actions, rewards, dones, log_probs):

        discounted_rewards = []
        discounted_reward = 0
        for i in reversed(range(len(rewards))):
            if dones[i] == True:
                discounted_reward = 0  
            discounted_reward = rewards[i] + self.gamma*discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        
        old_log_probs = torch.tensor(log_probs).squeeze()

        states = torch.tensor(states).squeeze().float()
        actions = torch.tensor(actions)

        for epoch in range(self.K_epochs):

            # for state in states:
            #     _, log_prob, state_value = self.policy.evaluate(torch.from_numpy(state).float().reshape(-1).unsqueeze(0))
            #     new_log_probs.append(log_prob)
            #     state_values = torch.stack(state_values, 1).squeeze()

            new_log_probs, state_values = self.policy.evaluate(states, actions)

            new_log_probs = new_log_probs.squeeze()
            advantages = discounted_rewards - state_values.squeeze()

            ratios = new_log_probs/old_log_probs
            ratios_clipped = torch.clamp(ratios, min=1-self.epsilon, max=1+self.epsilon)

            loss = -torch.min(ratios*advantages, ratios_clipped*advantages)

            self.optimizer.zero_grad()

            loss.mean().backward()
            self.optimizer.step()


            

env = gym.make('MountainCarContinuous-v0')
env.seed(0)
print('observation space:', env.observation_space)
print('action space:', env.action_space)
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
# agent = ActorCritic(state_size, action_size)
# action = agent.act(torch.from_numpy(state).float().unsqueeze(0))
# print(action)

n_episodes = 1000
max_steps = 1500
update_interval = 4000
log_interval = 20
time_step = 0

scores = deque(maxlen=log_interval)
episode_lengths = deque(maxlen=log_interval)

states = []
state_values = []
log_probs = []
actions =  []
rewards =  []
dones =  []

agent = PPO(env)

for n_episode in range(1, n_episodes+1):
    state = env.reset()

    episode_length = 0
    for t in range(max_steps):
        time_step += 1
        action, log_prob = agent.policy.act(torch.from_numpy(state).float().reshape(-1).unsqueeze(0))
        state, reward, done, _ = env.step(action.squeeze(0).numpy())
        actions.append(action)
        log_probs.append(log_prob)
        states.append(state)
        rewards.append(reward)
        dones.append(done)

        total_reward = sum(rewards)
        scores.append(total_reward)

        if time_step % update_interval == 0:
            agent.update(states, actions, rewards, dones, log_probs)
            time_step = 0
            states.clear()
            state_values.clear()
            log_probs.clear()
            actions.clear()
            rewards.clear()
            dones.clear()

        episode_length = t

        if done:
            break
    
    episode_lengths.append(episode_length)
    


    
    if n_episode % log_interval == 0:
        print("Episode: ", n_episode, "\t Avg. episode length: ", np.mean(episode_lengths), "\t Avg. score: ", np.mean(scores))
