# Environment and Task Description

This environment and task are derived from the following research by [Google Research](https://research.google/pubs/data-center-cooling-using-model-predictive-control/).


The goal of this environment is to address the problem of data center cooling by optimizing control strategies that reduce energy consumption while maintaining safe operating temperatures. It serves as a testbed for developing and evaluating reinforcement learning algorithms in real-world-inspired control scenarios.


Three RL agents (DQN, PPO, SAC) are used as a baseline.

For `offline training`, run the following commands:

```
python casestudy3/train_dqn.py
python casestudy3/train_ppo.py
python casestudy3/train_sac.py
```

After training, the model weights are saved to `. /model/`.

For online testing, run the following commands:
```
python casestudy3/test_dqn.py
python casestudy3/test_ppo.py
python casestudy3/test_sac.py
```