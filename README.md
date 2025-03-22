#  Discovery of Optimal Reward Function

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/facebookresearch/mtrl/blob/main/LICENSE)  [![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-371/)  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This repository contains the official implementation of the paper *"Discovery of the Optimal Reward Function for Embodied Reinforcement Learning Agents"*. 
It includes scripts and experiments designed to reproduce results from the research.
The paper is under peer review.

## Table of Contents

1. [Installation](#installation)
2. [Usage Instructions](#usage-instructions)
3. [Repository Structure](#repository-structure)
4. [Acknowledgements](#acknowledgements)

---

## Installation

### Prerequisites

- Python >= 3.7.1 and < 3.11
- Linux or Windows 10/11

### Setup

To ensure compatibility with the agents and to install the required dependencies, follow these steps:

```bash
git clone https://github.com/zhshao17/Discovery-of-Optimal-Reward-function.git
cd Discovery-of-Optimal-Reward-function
pip install -r requirements/requirements.txt
```

For running simulator tasks, the following optional dependencies are recommended:

```bash
# Optional dependencies
pip install -r requirements/requirements-mujoco.txt
pip install -r requirements/requirements-mujoco_py.txt
pip install -r requirements/requirements-envpool.txt
```

For energy management tasks, refer to the documentation from [Google Research](https://research.google/pubs/data-center-cooling-using-model-predictive-control/) and the [related README](casestudy3/README.md).

For UAV control tasks, you can install the `pyflyt` package via pip, or follow the setup instructions found [here](casestudy4/README.md):

```bash
pip3 install wheel numpy
pip3 install pyflyt
```

---

## Usage Instructions

### Running Experiments

To reproduce experiments, such as those from `casestudy1` with the `CartPole-v1` environment, you can execute the following commands:

```bash
python dqn.py --env_id CartPole-v1 --total_timesteps 500000
python ppo.py --env_id CartPole-v1 --total_timesteps 500000
```

Additional optional parameters for tuning experiments are outlined in the appendix of the our paper.

All experiments are run in parallel across five instances, with error metrics calculated for each.

---

## Repository Structure

The directory structure of this repository is as follows:

```
├── README.md                                          # This file
├── casestudy1                                         # Sparse reward tasks
│   ├── dqn.py                                         # DQN agent 
│   └── ppo.py                                         # PPO agent
├── casestudy2                                         # High-dimensional control tasks
│   ├── ppo_continuous_action.py                       # PPO agent
│   └── sac_continuous_action.py                       # SAC agent
├── casestudy3                                         # Energy management task
│   ├── README.md                                      # Task setup instructions
│   ├── test_ppo.py                                    # Online test for PPO agent 
│   ├── test_sac.py                                    # Online test for SAC agent 
│   ├── test_dqn.py                                    # Online test for DQN agent 
│   ├── train_dqn.py                                   # Offline training for DQN agent 
│   ├── train_ppo.py                                   # Offline training for PPO agent
│   └── train_sac.py                                   # Offline training for SAC agent
├── casestudy4                                         # Unmanned systems control task
│   ├── README.md                                      # Task setup instructions
│   ├── ppo_continuous_action.py                       # PPO agent
│   ├── sac_continuous_action.py                       # SAC agent
│   ├── td3_continuous_action.py                       # TD3 agent
│   └── test                                           # Test scripts
│       └── test_gym_envs.py                           # Environment test
├── cleanrl_utils                                      # Utility functions for agent construction
│   ├── ...                                            # Additional utilities
├── tests                                              # Test scripts
│   ├── ...                                            # Environment test
└── utils                                              # Miscellaneous utilities
    ├── agent.py                                       # Agent structure for casestudy3
    ├── data_center.py                                 # Data center settings for casestudy3
    ├── reward_machine.py                              # Reward function
    └── reward_model.py                                # Structure for reward function
```

---

## Acknowledgements

This code is adapted from the [CleanRL](https://github.com/vwxyzjn/cleanrl) repository and modified for the purposes of this research.
