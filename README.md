
[//]: # (Image References)

[image1]: https://video.udacity-data.com/topher/2018/June/5b1ea778_reacher/reacher.gif "Trained Agent"

# Udacity Continuous Control Project 2

### Introduction

The goal of this project is to train an agent to control a robotic arm, such that it stays close to a floating ball.

A reward of +0.1 is given for each time step that the arm is in the region of a floating ball.  The environment is considered to be "solved" when the agent receives a reward of > 30.0 for 100 consecutive episodes of 1000 steps.

![Trained Agent][image1]

### Environment Details

The environment is provided by [Unity](https://unity.com/), a company that specializes in building worlds that can be used for video game development, simulation, animation, and architecture/design.  The following is the description of the state space and actions available to the agent:

*The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.*

### Dependencies

1. Download the x64 windows environment for the single agent, from [here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip).

2. The code expects the Reacher.exe file to be located in the following directory of the repo  "./Reacher_Windows_x86_64_SingleAgent/Reacher.exe"

3. Create a new conda environment with the provided requirements.txt file. Ex. conda create --name <env> --file requirements.txt

## Using the Code

The code may be run using the command `python continuous_control.py <config.json file> [network_file.pth]`.  The network file is an optional parameter that will first load a previous file. 

### Train/Run Mode

Example `python navigator.py config.json`

The arguments to the program are provided using a .json file.  See the utilities/config.py file for the default parameters.  Setting `train_mode: true` will train the agent.  Setting `train_mode: false` will simply run the agent for a single episode without training the networks.

```python
{
    "env_name": "reacher",
    "train_mode": true,
    "device": "cuda",
    "actor_learning_rate": 1e-4,
    "critic_learning_rate": 1e-4,
    "batch_size": 8,
    "gamma": 0.99,
    "tau": 1e-3
}
```