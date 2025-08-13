
import os
import torch
import numpy as np
import random
import argparse
from collections import namedtuple
from utils import data_center
from utils import reward_machine
from utils.agent import SACD_agent


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train", default="True", help="Set this flag to enable training mode")
    parser.add_argument("--number_epochs", type=int, default=200)
    parser.add_argument("--state_dim", type=int, default=3)
    parser.add_argument("--action_dim", type=int, default=5)
    parser.add_argument("--temperature_step", type=float, default=1.5)
    parser.add_argument("--initial_month", type=int, default=0)
    parser.add_argument("--initial_users", type=int, default=20)
    parser.add_argument("--initial_rate_data", type=int, default=30)
    parser.add_argument("--hid_shape", type=int, nargs="+", default=[64, 64, 16])
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--adaptive_alpha", type=bool, default=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_memory", type=int, default=100000)
    
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--encode_dim", type=int, default=16)
    parser.add_argument("--reward_lr", type=float, default=1e-4)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--last_activation", type=str, default="None")
    parser.add_argument("--reward_buffer_size", type=int, default=100)
    parser.add_argument("--reward_n_samples", type=int, default=64)
    parser.add_argument("--n_samples", type=int, default=10)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1037)
    random.seed(1037)
    torch.manual_seed(1037)
    direction_boundary = (args.action_dim - 1) / 2
    env = data_center.Environment(
            optimal_temperature=(18.0, 24.0),
            initial_month=args.initial_month,
            initial_number_users=args.initial_users,
            initial_rate_data=args.initial_rate_data
        )
    env.train = args.train
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = SACD_agent(args, device)
    reward_function = reward_machine.RewardFunction(env=env, args=args, device=device)
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'log_probs', 'mu', 'overline_V'])
    if env.train:
        for epoch in range(1, args.number_epochs + 1):
            total_reward = 0
            new_month = np.random.randint(0, 12)
            env.reset(new_month=new_month)
            game_over = False
            current_state, _, _ = env.observe()
            timestep = 0
            epidata = []

            while not game_over and timestep <= 5 * 30 * 24 * 60:
                action, logprob, pi_s = agent.select_action(current_state, deterministic=False)

                direction = -1 if action < direction_boundary else 1
                energy_ai = abs(action - direction_boundary) * args.temperature_step

                next_state, _, game_over = env.update_env(direction, energy_ai, int(timestep / (30 * 24 * 60)))

                reward = reward_function.ovserve_reward(np.array(current_state), action, np.array(next_state)).item()

                agent.replay_buffer.add(np.array(current_state), action, reward, np.array(next_state), game_over)

                total_reward += reward
                current_state = next_state
                timestep += 1

                transition = Transition(state=np.array(current_state), action=action, reward=reward, log_probs=logprob, mu=pi_s, overline_V=0.0)
                epidata.append(transition)

                if agent.replay_buffer.size % args.batch_size == 0:
                    for _ in range(args.batch_size):
                        agent.train()

            epidata = reward_function.store_V(epidata)
            
            reward_function.D_xi.append(epidata)
            
            reward_function.optimize_reward(agent)

            agent.save()
