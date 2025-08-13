import os
import torch
import numpy as np
import random
import argparse
from utils import data_center
from utils.agent import SACD_agent, Policy_Net


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train", default=False, help="Set this flag to train the model")
    parser.add_argument("--model_path", type=str, default="./model/ppo+_actor.pth", help="Path to load SAC model")
    parser.add_argument("--number_actions", type=int, default=5)
    parser.add_argument("--temperature_step", type=float, default=1.5)
    parser.add_argument("--initial_month", type=int, default=0)
    parser.add_argument("--initial_users", type=int, default=20)
    parser.add_argument("--initial_rate_data", type=int, default=30)
    parser.add_argument("--state_dim", type=int, default=3)
    parser.add_argument("--hid_shape", type=int, nargs="+", default=[64, 64, 16])
    parser.add_argument("--simulation_time", type=int, default=12 * 30 * 24 * 60)

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    
    os.environ['PYTHONHASHSEED'] = '0'
    np.random.seed(1037)
    random.seed(1037)
    torch.manual_seed(1037)

    direction_boundary = (args.number_actions - 1) / 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    env = data_center.Environment(
        optimal_temperature=(18.0, 24.0),
        initial_month=args.initial_month,
        initial_number_users=args.initial_users,
        initial_rate_data=args.initial_rate_data
    )
    env.train = args.train

    
    model = Policy_Net(
        state_dim=args.state_dim,
        action_dim=args.number_actions,
        hid_shape=args.hid_shape
    ).to(device)

    print(f"Load model from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    current_state, _, _ = env.observe()

    for timestep in range(args.simulation_time):
        with torch.no_grad():
            current_state_tensor = torch.tensor(current_state, dtype=torch.float32).to(device)
            action_probs = model(current_state_tensor)
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample().item()

        direction = -1 if (action - direction_boundary < 0) else 1
        energy_ai = abs(action - direction_boundary) * args.temperature_step

        next_state, reward, game_over = env.update_env(
            direction, energy_ai, int(timestep / (30 * 24 * 60))
        )
        current_state = next_state
