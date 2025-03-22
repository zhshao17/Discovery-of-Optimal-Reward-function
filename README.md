[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/facebookresearch/mtrl/blob/main/LICENSE)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-371/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Discovery of Optimal Reward Function
<p align="center">
Official implementation of <strong>Discovery of the Optimal Reward Function for Embodied Reinforcement Learning Agents</strong> and scripts to reproduce experiments.</a>.
</p>


## Repository Agenda

1. [Install](#install)

2. [Instructions](#instructions)

3. [Repository Structure](#repository-structure)

4. [Acknowledgement](#acknowledgement)

## Install

Prerequisites:
* Python >=3.7.1, < 3.11
* Linux, Windows 10/11

To ensure compatibility with the agents, please install the packages listed in `requirements.txt`.

```
git clone https://github.com/zhshao17/Discovery-of-Optimal-Reward-function.git
cd Discovery-of-Optimal-Reward-function
pip install -r requirements/requirements.txt
```

For running simulator tasks, the following optional dependencies are recommended:

```
# optional dependencies
pip install -r requirements/requirements-mujoco.txt
pip install -r requirements/requirements-mujoco_py.txt
pip install -r requirements/requirements-envpool.txt
```

For running energy management tasks, task setup remains the same as [Google Research](https://research.google/pubs/data-center-cooling-using-model-predictive-control/), follow the documents in [here](casestudy3/README.md).

For running UAVs control tasks, you can install `pyflyt` by pip, or follow the documents in [here](casestudy4/README.md).

```
pip3 install wheel numpy
pip3 install pyflyt
```


# Instructions

Experiments:

In order to reproduce the experiments of casestudy1 with `cartpole-v1`, for example, you can enter in the `terminal`:

```
python dqn.py --env_id CartPole-v1 --total_timesteps 500000 
python ppo.py --env_id CartPole-v1 --total_timesteps 500000 
```

Other optional parameter settings are provided in detail in the appendix of the [paper]().

All experiments are executed in parallel five times and the errors are calculated.

## Repository Structure
The detailed structure of this project is shown as follows:

```
├── README.md                                          -- this file
├── casestudy1                                         -- sparse reward tasks
│   ├── dqn.py                                         -- DQN agent 
│   └── ppo.py                                         -- PPO agent
├── casestudy2                                         -- high-dimensional control tasks
│   ├── ppo_continuous_action.py                       -- PPO agent
│   └── sac_continuous_action.py                       -- SAC agent
├── casestudy3                                         -- energy management task
│   ├── README.md                                      -- file for construct task
│   ├── test ppo.py                                    -- online test for PPO agent 
│   ├── test sac.py                                    -- online test for SAC agent 
│   ├── test_dqn.py                                    -- online test for DQN agent 
│   ├── train dqn.py                                   -- offline train for DQN agent 
│   ├── train ppo.py                                   -- offline train for PPO agent
│   └── train sac.py                                   -- offline train for SAC agent
├── casestudy4                                         -- unmanned systems control Task
│   ├── README.md                                      -- file for construct task
│   ├── ppo_continuous_action.py                       -- PPO agent
│   ├── sac_continuous_action.py                       -- SAC agent
│   ├── td3_continuous_action.py                       -- TD3 agent
│   └── test                                           -- 
│       └── test_gym_envs.py                           -- env test
├── cleanrl_utils                                      -- 
│   ├── ...                                            -- utils for agent construction
└── utils                                              -- 
    ├── agent.py                                       -- agent structure for casestudy3
    ├── data_center.py                                 -- data center seeting for casestudy3
    ├── reward_machine.py                              -- reward function
    └── reward_model.py                                -- structure for reward function
```

## Acknowledgement
This code is originated and modified from <a href="https://github.com/vwxyzjn/cleanrl">CleanRL.
