# UAV Control task for Reinforcement Learning

The experiments are based on [`Pybullet`](https://pypi.org/project/pybullet/), and the UAV
model is implemented via `ArduPilot` and `PX4`.

## Installation

We recommend installations using Python [virtual environments](https://docs.python.org/3/library/venv.html).
It is possible to install PyFlyt using [`conda`](https://docs.conda.io/en/latest/), but YMMV.

### Linux and MacOS

Installation on _Linux_ and _MacOS_ is simple:
```sh
pip3 install wheel numpy
pip3 install pyflyt
```
> `numpy` and `wheel` must be installed prior to `pyflyt` such that `pybullet` is built with `numpy` support.

### Windows

1. First, install Microsoft Visual Studio Build Tools.
    - Go [here](https://visualstudio.microsoft.com/downloads/), scroll down to **All Downloads**, expand **Tools for Visual Studio**, download **Build Tools for Visual Studio 20XX**, where XX is just the latest year available.
    - Run the installer.
    - Select **Desktop development with C++**, then click **Install while downloading** or the alternate option if you wish.
2. Now, you can install `PyFlyt` as usual:
    ```sh
    pip3 install wheel numpy
    pip3 install pyflyt
    ```

### Gymnasium

```python
import gymnasium
import PyFlyt.gym_envs # noqa

env = gymnasium.make("PyFlyt/QuadX-Hover-v2", render_mode="human")
obs = env.reset()

termination = False
truncation = False

while not termination or truncation:
    observation, reward, termination, truncation, info = env.step(env.action_space.sample())
```

The official documentation for gymnasium environments is in [here](https://jjshoots.github.io/PyFlyt/documentation/gym_envs.html).


### Experiment

The task used for RL is `QuadX-Waypoints-v3`, run the follwing code in `terminal`:

```
python sac_continuous_action.py --env_id PyFlyt/QuadX-Waypoints-v3 --total_timesteps 1000000 
```