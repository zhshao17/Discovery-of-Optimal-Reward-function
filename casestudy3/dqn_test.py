import os
import torch
import numpy as np
import random
import argparse
from utils import data_center
from utils.agent import DQN, Brain


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", default=False, help="Set this flag to train the model")
    parser.add_argument("--model_path", type=str, default="./model/DQN+.pth", help="Path to load model")

    parser.add_argument("--number_actions", type=int, default=5, help="Number of possible actions")
    parser.add_argument("--temperature_step", type=float, default=1.5, help="Temperature step for energy calculation")
    parser.add_argument("--initial_month", type=int, default=0, help="Initial month for environment setup")
    parser.add_argument("--initial_users", type=int, default=20, help="Initial number of users in the environment")
    parser.add_argument("--initial_rate_data", type=int, default=30, help="Initial rate data for environment")

    parser.add_argument("--learning_rate", type=float, default=0.00001, help="Learning rate for model")

    parser.add_argument("--simulation_time", type=int, default=12 * 30 * 24 * 60, help="Total simulation time (in minutes)")

    return parser.parse_args()

    
    
if __name__ == "__main__":
    args = get_args()
    
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1037)
    random.seed(1037)
    torch.manual_seed(1037)
    direction_boundary = (args.number_actions - 1) / 2

    
    env = data_center.Environment(
        optimal_temperature=(18.0, 24.0),
        initial_month=args.initial_month,
        initial_number_users=args.initial_users,
        initial_rate_data=args.initial_rate_data
    )
    env.train = args.train
    
    
    model = Brain(args)
    model.load_state_dict(torch.load(args.model_path))  
    model.eval()  
    
    current_state, _, _ = env.observe()
    
    
    for timestep in range(args.simulation_time):
        current_state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
        
        
        with torch.no_grad():
            q_values = model(current_state_tensor)
        
        
        action = torch.argmax(q_values).item()
        
        
        if action - direction_boundary < 0:
            direction = -1
        else:
            direction = 1
        energy_ai = abs(action - direction_boundary) * args.temperature_step

        
        next_state, reward, game_over = env.update_env(direction, energy_ai, int(timestep / (30 * 24 * 60)))
        
        
        current_state = next_state