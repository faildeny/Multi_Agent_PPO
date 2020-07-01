import torch
from torch import nn
import torch.nn.functional as F
import gym
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
import numpy as np
from collections import deque

class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super().__init__()

        self.actor_fc1 = nn.Linear(state_size, 2*hidden_size)
        self.actor_fc2 = nn.Linear(2*hidden_size, hidden_size)

        self.actor_mu = nn.Linear(hidden_size, action_size)
        self.actor_sigma = nn.Linear(hidden_size, action_size)
        
        
        self.critic_fc1 = nn.Linear(state_size, 2*hidden_size)
        self.critic_fc2 = nn.Linear(2*hidden_size, hidden_size)

        self.critic_value = nn.Linear(hidden_size, 1)

        self.distribution = torch.distributions.Normal

        action_std = 0.5
        self.action_var = torch.full((action_size,), action_std*action_std)

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
        action_mu, action_sigma, state_value = self.forward(state)
        # m = self.distribution(action_mu, action_sigma)

        action_var = self.action_var.expand_as(action_mu)
        # print(action_var.shape)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mu, cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, state_value
    
    def evaluate(self, state, action):
        action_mu, action_sigma, state_value = self.forward(state)
        m = self.distribution(action_mu.squeeze(), action_sigma.squeeze())
        log_prob = m.log_prob(action)

        return log_prob, state_value

    def evaluate3(self, state, action):
        action_mu, action_sigma, state_value = self.forward(state)
        m = self.distribution(action_mu, 10*action_sigma)
        log_prob = m.log_prob(action.unsqueeze(1))

        return log_prob, state_value.squeeze(1)
    
    def evaluate2(self, state, action):   
        action_mean, _, state_value = self.forward(state)
        action_var = self.action_var.expand_as(action_mean)
        # print(action_var.shape)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # print(action.unsqueeze(1).shape)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        # state_value = self.critic(state)
        # print(action_logprobs.shape)

        # print("After evaluate")

        
        return action_logprobs, torch.squeeze(state_value)#, dist_entropy


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

        self.MseLoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=(0.9, 0.999))

    def update(self, states, actions, rewards, dones, log_probs, state_values):

        discounted_rewards = []
        discounted_reward = 0
        for i in reversed(range(len(rewards))):
            if dones[i] == True:
                discounted_reward = 0  
            discounted_reward = rewards[i] + self.gamma*discounted_reward
            discounted_rewards.insert(0, discounted_reward)
        
        discounted_rewards = torch.tensor(discounted_rewards)
        old_state_values = torch.stack(state_values, 1).detach()
        advantages = discounted_rewards - old_state_values.detach().squeeze()
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        

        states = torch.tensor(states).squeeze().float()
        # print(actions.size)
        # print(actions[0])
        # actions = torch.tensor(actions)
        # old_log_probs = torch.tensor(log_probs).squeeze()
        
        # states = torch.squeeze(torch.stack(states), 1).detach()
        actions = torch.squeeze(torch.stack(actions), 1).detach()
        old_log_probs = torch.squeeze(torch.stack(log_probs), 1).detach()

        for epoch in range(self.K_epochs):
            # print("inside training loop")
            # for state in states:
            #     _, log_prob, state_value = self.policy.evaluate(torch.from_numpy(state).float().reshape(-1).unsqueeze(0))
            #     new_log_probs.append(log_prob)
            #     state_values = torch.stack(state_values, 1).squeeze()
            new_log_probs, state_values = self.policy.evaluate2(states, actions)
            
            # print(actions.shape)
            # print(states.shape)
            # print(state_values.shape)
            # print(discounted_rewards.shape)

            new_log_probs = new_log_probs.squeeze()

            # advantages = discounted_rewards - state_values.detach().squeeze()
            
            # print("New log porbs: ", new_log_probs.shape)
            # print("Old log porbs: ", old_log_probs.shape)

            ratios = torch.exp(new_log_probs - old_log_probs.detach())
            ratios_clipped = torch.clamp(ratios, min=1-self.epsilon, max=1+self.epsilon)

            loss = -torch.min(ratios*advantages, ratios_clipped*advantages)+ 0.5*self.MseLoss(state_values, discounted_rewards)

            self.optimizer.zero_grad()

            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

            



n_episodes = 10000
max_steps = 1500
update_interval = 4000
log_interval = 20
time_step = 0
solving_threshold = 300

render = False
train = True
pretrained = True

# env_name = 'MountainCarContinuous-v0'
env_name = "BipedalWalker-v3"
env = gym.make(env_name)
env.seed(0)
print('observation space:', env.observation_space)
print('action space:', env.action_space)
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
# agent = ActorCritic(state_size, action_size)
# action = agent.act(torch.from_numpy(state).float().unsqueeze(0))
# print(action)

scores = deque(maxlen=log_interval)
max_score = -1000
episode_lengths = deque(maxlen=log_interval)

states = []
state_values = []
log_probs = []
actions =  []
rewards =  []
dones =  []

agent = PPO(env)

if not train:
    # agent.policy_old.load_state_dict(torch.load('./PPO_model_solved_'+env_name+'.pth'))
    agent.policy_old.eval()

if pretrained:
    agent.policy_old.load_state_dict(torch.load('./PPO_model_best_'+env_name+'.pth'))
    agent.policy.load_state_dict(torch.load('./PPO_model_best_'+env_name+'.pth'))


for n_episode in range(1, n_episodes+1):
    state = env.reset()

    episode_length = 0
    for t in range(max_steps):
        time_step += 1

        # state = torch.FloatTensor(state.reshape(1, -1))
        
        action, log_prob, state_value = agent.policy_old.act(torch.from_numpy(state).float().reshape(-1).unsqueeze(0))
        # action, log_prob = agent.policy_old.act(state)
        state, reward, done, _ = env.step(action.squeeze(0).numpy())

        # state = torch.FloatTensor(state.reshape(1, -1))
        rewards.append(reward)
        
        if render:
            env.render()

        if train:

            actions.append(action)
            log_probs.append(log_prob)
            states.append(state)
            dones.append(done)
            state_values.append(state_value)

            if time_step % update_interval == 0:
                # print("Updating agent")
                # print("Episode: ", n_episode)
                agent.update(states, actions, rewards, dones, log_probs, state_values)
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
    total_reward = sum(rewards[-episode_length:])
    # print("Episode: ", n_episode, "\t Episode length: ", episode_length, "\t Score: ", total_reward)
    scores.append(total_reward)
    total_reward = 0

    if train: 
        if n_episode % log_interval == 0:
            print("Episode: ", n_episode, "\t Avg. episode length: ", np.mean(episode_lengths), "\t Avg. score: ", np.mean(scores))

            if np.mean(scores) > solving_threshold:
                print("Environment solved, saving model")
                torch.save(agent.policy_old.state_dict(), 'PPO_model_solved_{}.pth'.format(env_name))
            
            if np.mean(scores) > max_score:
                print("Saving improved model")

                max_score = np.mean(scores)
                torch.save(agent.policy_old.state_dict(), 'PPO_model_best_{}.pth'.format(env_name))

        # if n_episode % 300 == 0:
        #     print("Saving model")
        #     torch.save(agent.policy_old.state_dict(), 'PPO_model_{}_epoch_{}.pth'.format(env_name, n_episode))