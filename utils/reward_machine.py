import torch
import torch.nn as nn
import os
import numpy as np
from torch.autograd import Variable
from reward_model import Critic, Reward
import torch.optim as optim
from collections import deque


class RewardFunction():
    def __init__(self, env, args, device):
        activation_function_list = {
            "relu": torch.relu,
            "sigmoid": torch.sigmoid,
            "tanh": torch.tanh,
            "None": None
        }
        self.hidden_dim = args.hidden_dim
        self.encode_dim = args.encode_dim
        self.gamma = args.gamma
        self.lr = args.reward_lr
        self.activate_function = activation_function_list[args.activate_function]
        self.last_activate_function = activation_function_list[args.last_activate_function]
        self.device = device
        self.state_dim = env.observation_space.shape
        self.action_dim = env.action_space.shape
        self.value_function = Critic(layer_num=3, input_dim=self.state_dim, output_dim=1,\
                                    hidden_dim=self.hidden_dim,
                                    activation_function=self.activate_function, last_activation = None).to(device=self.device)
        self.value_function_optimizer = optim.Adam(self.value_function.parameters(), lr=self.lr)
        self.reward_function = Reward(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim, \
                                     encode_dim=self.encode_dim,\
                                     output_dim=1,\
                                     activation_function=self.activate_function,\
                                     last_activation=self.last_activate_function)\
                                    .to(self.device)
        self.reward_function_optimizer = optim.Adam(self.reward_function.parameters(), lr=self.lr)
        self.n_samples = args.n_samples
        self.D_xi = deque(maxlen=args.reward_buffer_size)

    def ovserve_reward(self, state, action, next_state=None):
        state = torch.Tensor(state).to(self.device)
        action = torch.Tensor(action).to(self.device)
        reward = self.reward_function.forward(state, action).detach().cpu().numpy()
        return reward

    def optimaze_reward(self, agent):
        D_new = [step for traj in self.D_xi for step in traj]
        np.random.shuffle(D_new)
        states_batch, overline_V_batch = [], []
        accumulator = []
        for step in D_new:
            s, a, reward_hat, log_probs, mu, overline_V = step
            states_batch.append(s)
            overline_V_batch.append(overline_V)
            prob_a = torch.exp(log_probs).to(self.device)
            V_s = self.value_function(torch.Tensor(s).to(self.device)).detach().cpu().item()
            action_bs, log_probs_action_bs = agent.get_action_prob_from_mu(mu, self.n_samples)
            s_expanded = torch.tensor(np.tile(s, (self.n_samples, 1)), dtype=torch.float32, device=self.device)
            reward_bs = self.reward_function(s_expanded, action_bs)
            probs_action_bs = torch.exp(log_probs_action_bs)
            E_R_b = torch.sum(probs_action_bs * reward_bs, dim=0)
            accumulator.append(prob_a * (overline_V - V_s) * (torch.Tensor(reward_hat).to(self.device) - E_R_b))
        self.optimize_value_function(np.array(states_batch), np.array(overline_V_batch))
        loss = torch.mean(torch.stack(accumulator))
        self.reward_function_optimizer.zero_grad()
        loss.backward()
        self.reward_function_optimizer.step()


    def optimize_value_function(self, states_batch, overline_V_batch):
        states_batch = torch.Tensor(states_batch).to(self.device)
        overline_V_batch = torch.Tensor(overline_V_batch).to(self.device)
        pred_batch = self.value_function.forward(states_batch)
        loss = torch.nn.functional.smooth_l1_loss(pred_batch, overline_V_batch.unsqueeze(1))
        self.value_function_optimizer.zero_grad()
        loss.backward()
        self.value_function_optimizer.step()

    def store_V(self, epidata):
        new_epidata = []
        for step in reversed(epidata):
            overline_V = step.reward + self.gamma * overline_V
            updated_step = step._replace(overline_V=overline_V)
            new_epidata.insert(0, updated_step)
        return new_epidata
