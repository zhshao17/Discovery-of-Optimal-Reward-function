
import os
import numpy as np
import random
from collections import namedtuple
from utils import data_center
from utils import reward_machine
from utils.agent import PPO_agent
import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--train", default="True", help="Set this flag to enable training mode")
    parser.add_argument("--number_epochs", type=int, default=200)
    parser.add_argument("--state_dim", type=int, default=3)
    parser.add_argument("--action_dim", type=int, default=5)
    parser.add_argument("--net_width", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lambd", type=float, default=0.95)
    parser.add_argument("--clip_rate", type=float, default=0.2)
    parser.add_argument("--K_epochs", type=int, default=10)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--max_memory", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epsilon", type=float, default=0.3)
    parser.add_argument("--number_actions", type=int, default=5)
    parser.add_argument("--temperature_step", type=float, default=1.5)
    
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--encode_dim", type=int, default=16)
    parser.add_argument("--reward_lr", type=float, default=1e-4)
    parser.add_argument("--activate_function", type=str, default="relu")
    parser.add_argument("--last_activate_function", type=str, default="None")
    parser.add_argument("--reward_buffer_size", type=int, default=100)
    parser.add_argument("--n_samples", type=int, default=10)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1037)
    random.seed(1037)
    torch.manual_seed(1037)
    direction_boundary = (args.number_actions - 1) / 2

    
    env = data_center.Environment(optimal_temperature = (18.0, 24.0), initial_month = 0, initial_number_users = 20, initial_rate_data = 30)
    
    env.train = args.train

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ppo = PPO_agent(args, device)

    reward_function = reward_machine.RewardFunction(env=env, args=args, device=device)
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'log_probs', 'mu', 'overline_V'])
    if (env.train):
        
        for epoch in range(1, args.number_epochs):
            
            total_reward = 0
            loss = 0.
            new_month = np.random.randint(0, 12)
            env.reset(new_month = new_month)
            game_over = False
            current_state, _, _ = env.observe()
            timestep = 0

            epidata = []
            
            while ((not game_over) and timestep <= 5 * 30 * 24 * 60):
                action, logprob, mu = ppo.select_action(current_state)
                
                direction = -1 if action < direction_boundary else 1
                energy_ai = abs(action - direction_boundary) * args.temperature_step

                
                next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30 * 24 * 60)))
                reward = reward_function.get_reward(np.array(current_state), action, np.array(next_state)) 
                transition = Transition(state=np.array(current_state), action=action, reward=reward, log_probs= logprob, mu=mu, overline_V=0.0)
                epidata.append(transition)

                ppo.store_transition(np.array(current_state), action, reward, np.array(next_state), logprob, game_over)
                total_reward += reward

                current_state = next_state
                timestep += 1

            ppo.train()
            
            epidata = reward_function.store_V(epidata)
            
            reward_function.D_xi.append(epidata)
            
            reward_function.optimize_reward(agent=ppo)

            
            ppo.save()


