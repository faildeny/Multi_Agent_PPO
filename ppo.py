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
        # print(state)
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
        return action
    
    def evaluate(self, state):
        action_mu, action_sigma, state_value = self.forward(state)
        m = self.distribution(action_mu, action_sigma)
        action = m.sample()
        log_prob = m.log_prob(action)

        return action, log_prob, state_value


class PPO():
    def __init__(self, env):

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.lr = 3e-4
        self.gamma  = 0.99
        self.epsilon = 0.2
        self.K_epochs = 80

        self.policy = ActorCritic(self.state_size, self.action_size)
        self.policy_old = ActorCritic(self.state_size, self.action_size)

        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def update(self, states, actions, rewards, dones, state_values, log_probs):

        discounted_rewards = []
        discounted_reward = 0
        for i in reversed(range(len(rewards))):
            if dones[i] == True:
                discounted_reward = 0  
            discounted_reward = rewards[i] + self.gamma*discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        state_values = torch.stack(state_values, 1).squeeze()
        advantages = discounted_rewards - state_values
        
        # old_log_probs = torch.stack(log_probs, 1).squeeze()
        old_log_probs = torch.tensor(log_probs).squeeze()
        

        for _ in range(self.K_epochs):
            new_log_probs = []

            for state in states:
                _, log_prob, _ = self.policy.evaluate(torch.from_numpy(state).float().reshape(-1).unsqueeze(0))
                new_log_probs.append(log_prob)
            
            _, log_prob, _ = self.policy.evaluate(torch.from_numpy(state).float().reshape(-1).unsqueeze(0))

            new_log_probs = torch.stack(new_log_probs, 1).squeeze()

            # print(state_values)
            # print(discounted_rewards)
            # print(new_log_probs)
            # print(old_log_probs)
            ratios = new_log_probs/old_log_probs
            ratios_clipped = torch.clamp(ratios, min=1-self.epsilon, max=1+self.epsilon)
            loss = -ratios_clipped*advantages

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

n_episodes = 200
max_steps = 1000
update_interval = 100
time_step = 0

scores = deque(maxlen=100)

states = []
state_values = []
log_probs = []
actions =  []
rewards =  []
dones =  []

agent = PPO(env)

for n_episode in range(1, n_episodes+1):
    log_probs = []
    state = env.reset()

    for t in range(max_steps):
        time_step += 1
        action, log_prob, state_value = agent.policy.evaluate(torch.from_numpy(state).float().reshape(-1).unsqueeze(0))
        state, reward, done, _ = env.step(action.numpy())
        
        actions.append(action)
        state_values.append(state_value)
        print(log_prob)
        log_probs.append(log_prob)

        states.append(state)
        rewards.append(reward)
        dones.append(done)

        total_reward = sum(rewards)
        scores.append(total_reward)

        if time_step % update_interval == 0:
            print("Time step: ",time_step)
            print(len(states))
            print(len(state_values))
            print(len(log_probs))
            print(len(actions))
            print(len(rewards))
            print(len(dones))

            print(state)
            print(log_prob)
            agent.update(states, actions, rewards, dones, state_values, log_probs)
            time_step = 0
            states.clear()
            state_values.clear()
            log_probs.clear()
            actions.clear()
            rewards.clear()
            dones.clear()
    

        if done:
            break

    print("Episode passed")
    print(len(states))
    print(len(state_values))
    print(len(log_probs))
    print(len(actions))
    print(len(rewards))
    print(len(dones))
    print("Time step: ",time_step)


    
    if n_episode % 10 == 0:
        print("Episode: ", n_episode, "Avg. score: ", np.mean(scores))

    # def act(self, state):
        #To be implemented