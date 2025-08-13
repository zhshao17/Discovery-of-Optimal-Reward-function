import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import argparse
from collections import namedtuple
from utils.data_center import Environment
from utils import reward_machine
from utils.agent import DQN, Brain


def get_args():
    parser = argparse.ArgumentParser()

    
    parser.add_argument("--train", default=True)
    parser.add_argument("--number_epochs", type=int, default=200)
    parser.add_argument("--epsilon", type=float, default=0.0)
    
    parser.add_argument("--number_actions", type=int, default=5)
    parser.add_argument("--temperature_step", type=float, default=1.5)
    parser.add_argument("--initial_month", type=int, default=0)
    parser.add_argument("--initial_users", type=int, default=20)
    parser.add_argument("--initial_rate_data", type=int, default=30)
    
    parser.add_argument("--state_dim", type=int, default=3)
    parser.add_argument("--action_dim", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--max_memory", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)

    
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--encode_dim", type=int, default=16)
    parser.add_argument("--reward_lr", type=float, default=1e-4)
    parser.add_argument("--activate_function", type=str, default="relu")
    parser.add_argument("--last_activate_function", type=str, default="None")
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--reward_buffer_size", type=int, default=100)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1037)
    random.seed(1037)
    torch.manual_seed(1037)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    direction_boundary = (args.number_actions - 1) / 2
    
    env = Environment(
        optimal_temperature=(18.0, 24.0),
        initial_month=args.initial_month,
        initial_number_users=args.initial_users,
        initial_rate_data=args.initial_rate_data
    )
    env.train = args.train
    
    brain = Brain(args)
    dqn = DQN(args)

    # Reward Setup
    reward_function = reward_machine.RewardFunction(env=env,args = args,device=device)
    Transition = namedtuple('Transition', ['state', 'action', 'reward', 'log_probs', 'mu', 'overline_V'])
    
    model_path = "./model/DQN+.pth"
    if env.train:
        for epoch in range(1, args.number_epochs + 1):
            total_reward = 0
            loss = 0.
            new_month = np.random.randint(0, 12)
            env.reset(new_month=new_month)
            current_state, _, _ = env.observe()
            game_over = False
            timestep = 0
            epidata = []

            while not game_over and timestep <= 5 * 30 * 24 * 60:
                
                if np.random.rand() <= args.epsilon:
                    action = np.random.randint(0, args.number_actions)
                else:
                    with torch.no_grad():
                        current_state_tensor = torch.tensor(current_state, dtype=torch.float32)
                        q_values = brain(current_state_tensor.unsqueeze(0))
                        action = torch.argmax(q_values).item()
                        log_prob = F.log_softmax(q_values, dim=1)

                direction = -1 if action < direction_boundary else 1
                energy_ai = abs(action - direction_boundary) * args.temperature_step

                next_state, _, game_over = env.update_env(
                    direction, energy_ai, int(timestep / (30 * 24 * 60))
                )

                reward = reward_function.ovserve_reward(
                    np.array(current_state), action, np.array(next_state), reward=None
                )

                with torch.no_grad():
                    q_values = brain(torch.tensor(current_state, dtype=torch.float32).unsqueeze(0))
                    transition = Transition(state=np.array(current_state), action=action, reward=reward, log_probs=log_prob, mu=q_values.numpy().reshape(-1), overline_V=0.0)
                    epidata.append(transition)

                total_reward += reward

                
                dqn.remember([current_state, action, reward, next_state], game_over)

                
                if len(dqn.memory) >= args.batch_size:
                    inputs, targets = dqn.get_batch(brain, batch_size=args.batch_size)

                    inputs_tensor = torch.tensor(inputs, dtype=torch.float32)
                    targets_tensor = torch.tensor(targets, dtype=torch.float32)

                    brain.optimizer.zero_grad()
                    predictions = brain(inputs_tensor)
                    batch_loss = brain.loss_fn(predictions, targets_tensor)
                    loss += batch_loss.item()
                    batch_loss.backward()
                    brain.optimizer.step()

                timestep += 1
                current_state = next_state

            
            epidata = reward_function.store_V(epidata)
            
            reward_function.D_xi.append(epidata)
            
            reward_function.optimize_reward(agent=dqn)

            
            torch.save(brain.state_dict(), model_path)