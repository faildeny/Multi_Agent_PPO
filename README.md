# PPO implementation in Pytorch

## General info

Implementation has been written as an exercise while exploring Reinforcement Learning concepts. 
The algorithm is based on the description provided in original [Proximal Policy Optimization paper](https://arxiv.org/abs/1707.06347) by OpenAI. However, to get a working version of algorithm, important implementation details were added from [The 32 Implementation Details of Proximal Policy Optimization (PPO) Algorithm](https://costa.sh/blog-the-32-implementation-details-of-ppo.html) and this [implementation](https://github.com/nikhilbarhate99/PPO-PyTorch).

## Dependencies
All dependencies are provided in `requirements.txt` file.
The implementation uses Pytorch for training and Gym for environments. The imageio is an optional dependency needed to save GIFs from rendered environment. 



## Example
Example of a trained agent in 'BipedalWalker-v3' environment after 2000 episodes.
![](images/walker_2000.gif)