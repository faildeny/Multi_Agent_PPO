from unityagents import UnityEnvironment
import torch
import numpy as np
from collections import deque
import time
import imageio
from torch.utils.tensorboard import SummaryWriter
from ppo import PPO, MemoryBuffer

env_name = "Reacher"

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

env = UnityEnvironment(file_name='../Reacher_Windows_x86_64_twenty/Reacher.exe', no_graphics=False)
brain_name = env.brain_names[0]
print("Brain name: ",env.brain_names)
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]


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
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    state = states[0]
    state = torch.FloatTensor(state.reshape(1, -1))

    episode_length = 0
    for t in range(max_steps):
        time_step += 1

        action, log_prob = agent.select_action(state)
        

        state = torch.FloatTensor(state.reshape(1, -1))

        memory.states.append(state)
        memory.actions.append(action)
        memory.logprobs.append(log_prob)

        actions = []
        ## Unity env style
        for agent_id in range(0,20):
            actions.append(action.data.numpy().flatten())

        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        
        states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done   
        
        state = states[0]
        reward = rewards[0]
        done = dones[0]

        # state, reward, done, _ = env.step(action.data.numpy().flatten())


        memory.rewards.append(reward)
        memory.dones.append(done)
        rewards.append(reward)
        state_value = 0
        
        # if render:
        #     image = env.render(mode = 'rgb_array')
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
            writer.add_scalars('Score', {'Score':total_reward, 'Avg._Score': np.mean(scores)}, n_episode)
            writer.add_scalars('Episode_length', {'Episode_length':episode_length, 'Avg._Episode length': np.mean(episode_lengths)}, n_episode)
    
    else:
        print("Episode: ", n_episode, "\t Episode length: ", episode_length, "\t Score: ", total_reward)
        
    total_reward = 0
