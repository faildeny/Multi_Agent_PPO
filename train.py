import gym
import torch
import numpy as np
from collections import deque
import time
import imageio
from torch.utils.tensorboard import SummaryWriter
from ppo import PPO, MemoryBuffer

env_name = "BipedalWalker-v3"

n_episodes = 1000
max_steps = 1600
update_interval = 4000
log_interval = 20
solving_threshold = 300
time_step = 0

render = False
train = True
pretrained = False
tensorboard_logging = True

env = gym.make(env_name)
env.seed(0)
print('observation space:', env.observation_space)
print('action space:', env.action_space)
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]

scores = deque(maxlen=log_interval)
max_score = -1000
episode_lengths = deque(maxlen=log_interval)
rewards =  []

memory = MemoryBuffer()

agent = PPO(state_size, action_size)

if not train:
    agent.policy_old.eval()
else:
    writer = SummaryWriter(log_dir='logs/'+env_name+'_'+str(time.time()))

if pretrained:
    agent.policy_old.load_state_dict(torch.load('./PPO_modeldebug_best_'+env_name+'.pth'))
    agent.policy.load_state_dict(torch.load('./PPO_modeldebug_best_'+env_name+'.pth'))

writerImage = imageio.get_writer('./images/run.gif', mode='I', fps=25)

for n_episode in range(1, n_episodes+1):
    state = env.reset()
    state = torch.FloatTensor(state.reshape(1, -1))

    episode_length = 0
    for t in range(max_steps):
        time_step += 1

        action, log_prob = agent.select_action(state, memory)
        

        state = torch.FloatTensor(state.reshape(1, -1))

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(log_prob)

        state, reward, done, _ = env.step(action.data.numpy().flatten())


        memory.rewards.append(reward)
        memory.dones.append(done)
        rewards.append(reward)
        state_value = 0
        
        if render:
            image = env.render(mode = 'rgb_array')
            # if time_step % 2 == 0:
            #     writerImage.append_data(image)

        if train:
            if time_step % update_interval == 0:
                agent.update(memory)
                time_step = 0
                memory.clear_buffer()

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

        if tensorboard_logging:
            writer.add_scalars('Score', {'Score':total_reward, 'Avg. Score': np.mean(scores)}, n_episode)
            writer.add_scalars('Episode length', {'Episode length':episode_length, 'Avg. Episode length': np.mean(episode_lengths)}, n_episode)
    
    else:
        print("Episode: ", n_episode, "\t Episode length: ", episode_length, "\t Score: ", total_reward)
        
    total_reward = 0
