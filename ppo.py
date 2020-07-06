import torch
from torch import nn
import torch.nn.functional as F
import gym
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from collections import deque
import time
import imageio

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
        action_mu, action_sigma, state_value = self.forward(state)
        # m = self.distribution(action_mu, action_sigma)

        action_var = self.action_var.expand_as(action_mu)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mu, cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(log_prob)

        return action.detach()#, log_prob, state_value
    
    def evaluate5(self, state, action):
        action_mu, action_sigma, state_value = self.forward(state)
        m = self.distribution(action_mu.squeeze(), action_sigma.squeeze())
        log_prob = m.log_prob(action)

        return log_prob, state_value

    def evaluate(self, state, action):   
        action_mean, _, state_value = self.forward(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.state_values = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.state_values[:]


class PPO():
    def __init__(self, env):

        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        self.lr = 0.0003
        self.gamma  = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 80

        self.policy = ActorCritic(self.state_size, self.action_size, 0.5)
        self.policy_old = ActorCritic(self.state_size, self.action_size, 0.5)

        self.MseLoss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=(0.9, 0.999))

    def update(self, memory):

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
            ratios_clipped = torch.clamp(ratios, min=1-self.eps_clip, max=1+self.eps_clip)
            loss = -torch.min(ratios*advantages, ratios_clipped*advantages)+ 0.5*self.MseLoss(state_values, discounted_rewards) - 0.01*dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())

    def select_action(self, state, memory):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.policy_old.act(state, memory).cpu().data.numpy().flatten()
    


n_episodes = 10000
max_steps = 1500
update_interval = 4000
log_interval = 20
time_step = 0
solving_threshold = 300

render = False
train = True
pretrained = False

# env_name = 'MountainCarContinuous-v0'
env_name = "BipedalWalker-v3"
env = gym.make(env_name)
env.seed(0)
print('observation space:', env.observation_space)
print('action space:', env.action_space)
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

scores = deque(maxlen=log_interval)
max_score = -1000
episode_lengths = deque(maxlen=log_interval)

states = []
state_values = []
log_probs = []
actions =  []
rewards =  []
dones =  []
memory = Memory()

agent = PPO(env)

if not train:
    # agent.policy_old.load_state_dict(torch.load('./PPO_model_solved_'+env_name+'.pth'))
    agent.policy_old.eval()
else:
    writer = SummaryWriter(log_dir='logs/'+env_name+'_'+str(time.time()))


if pretrained:
    agent.policy_old.load_state_dict(torch.load('./PPO_model_best_'+env_name+'.pth'))
    agent.policy.load_state_dict(torch.load('./PPO_model_best_'+env_name+'.pth'))

# with imageio.get_writer('./videos/run.gif', mode='I', fps=50) as writer:

for n_episode in range(1, n_episodes+1):
    state = env.reset()
    state = torch.FloatTensor(state.reshape(1, -1))

    episode_length = 0
    for t in range(max_steps):
        time_step += 1

        action = agent.select_action(state, memory)
        
        state, reward, done, _ = env.step(action)

        state = torch.FloatTensor(state.reshape(1, -1))

        memory.rewards.append(reward)
        memory.dones.append(done)
        rewards.append(reward)
        state_value = 0
        
        if render:
            env.render()
            # image = env.render(mode = 'rgb_array')
            # writer.append_data(image)

        if train:
            if time_step % update_interval == 0:
                agent.update(memory)
                time_step = 0
                memory.clear_memory()

        episode_length = t

        if done:
            break
    
    episode_lengths.append(episode_length)
    total_reward = sum(memory.rewards[-episode_length:])
    scores.append(total_reward)
    
    if train:
        if n_episode % log_interval == 0:
            print("Episode: ", n_episode, "\t Avg. episode length: ", np.mean(episode_lengths), "\t Avg. score: ", np.mean(scores))

            if np.mean(scores) > solving_threshold:
                print("Environment solved, saving model")
                torch.save(agent.policy_old.state_dict(), 'PPO_model_solved_{}.pth'.format(env_name))
            
        if total_reward > max_score:
            print("Saving improved model")

            max_score = total_reward
            torch.save(agent.policy_old.state_dict(), 'PPO_modeldebug_best_{}.pth'.format(env_name))
        
        writer.add_scalars('Score', {'Score':total_reward, 'Avg. Score': np.mean(scores)}, n_episode)
        writer.add_scalars('Episode length', {'Episode length':episode_length, 'Avg. Episode length': np.mean(episode_lengths)}, n_episode)
    
    total_reward = 0

writer.close()
