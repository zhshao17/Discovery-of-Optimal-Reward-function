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
        """
        Initialize the reward function used for upper-level optimization

        Args:
            env: Gym-like environment
            args: 
            device: "cpu" or "cuda"
        """
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
        # Value function V(s) used to approximate standard state values
        self.value_function = Critic(layer_num=3, input_dim=self.state_dim, output_dim=1,\
                                    hidden_dim=self.hidden_dim,
                                    activation_function=self.activate_function, last_activation = None).to(device=self.device)
        self.value_function_optimizer = optim.Adam(self.value_function.parameters(), lr=self.lr)
        # Reward function R(s, a)
        self.reward_function = Reward(state_dim=self.state_dim, action_dim=self.action_dim, hidden_dim=self.hidden_dim, \
                                     encode_dim=self.encode_dim,\
                                     output_dim=1,\
                                     activation_function=self.activate_function,\
                                     last_activation=self.last_activate_function)\
                                    .to(self.device)
        self.reward_function_optimizer = optim.Adam(self.reward_function.parameters(), lr=self.lr)
        self.D_xi = deque(maxlen=args.reward_buffer_size) # Trajectory buffer
        self.n_samples = args.n_samples # Number of sampled actions for estimating action set

    def ovserve_reward(self, state, action, next_state=None):
        """
        Give the current reward function on a given state-action pair
        Args:
            state: Current state
            action: Taken action
            next_state (optional): Next state (unused)

        Returns:
            Reward signal for the state-action pair
        """
        state = torch.Tensor(state).to(self.device)
        action = torch.Tensor(action).to(self.device)
        reward = self.reward_function.forward(state, action).detach().cpu().numpy()
        return reward

    def optimize_reward(self, agent):
        """
        Perform the upper-level optimization step to update the reward function
        
        """
        # decompose trajectories in D into individual transitions
        D_new = [step for traj in self.D_xi for step in traj]
        np.random.shuffle(D_new)
        states_batch, overline_V_batch = [], []
        accumulator_1, accumulator_2 = [], []
        for step in D_new: 
            s, a, reward_hat, log_probs, mu, overline_V = step # tau
            states_batch.append(s)
            overline_V_batch.append(overline_V)
            # compute π(a|s) using the policy distribution
            prob_a = torch.exp(log_probs).to(self.device)
            # estimate state value V(s)
            V_s = self.value_function(torch.Tensor(s).to(self.device)).detach().cpu().item()
            # compute the first term of the gradient
            accumulator_2.append(prob_a * (overline_V - V_s))
            # sample actions from action space
            action_bs, log_probs_action_bs = agent.get_action_prob_from_mu(mu, self.n_samples)
            # compute reward for sampled actions
            s_expanded = torch.tensor(np.tile(s, (self.n_samples, 1)), dtype=torch.float32, device=self.device)
            reward_bs = self.reward_function(s_expanded, action_bs)
            # estimate the reward center
            probs_action_bs = torch.exp(log_probs_action_bs)
            reward_center = torch.sum(probs_action_bs * reward_bs, dim=0)
            # compute the second term of the gradient
            accumulator_1.append(torch.Tensor(reward_hat).to(self.device) - reward_center)
        self.optimize_value_function(np.array(states_batch), np.array(overline_V_batch))
        loss = torch.mean(torch.stack(accumulator_2)) * torch.mean(torch.stack(accumulator_1))
        # update the reward function
        self.reward_function_optimizer.zero_grad()
        loss.backward()
        self.reward_function_optimizer.step()

    def optimize_value_function(self, states_batch, overline_V_batch):
        """
        Optimize the value function V(s) to regress toward the standardized return overline_V
        
        """
        # array --> tensor
        states_batch = torch.Tensor(states_batch).to(self.device)
        overline_V_batch = torch.Tensor(overline_V_batch).to(self.device)
        pred_batch = self.value_function.forward(states_batch) # [B, 1]
        loss = torch.nn.functional.smooth_l1_loss(pred_batch, overline_V_batch.unsqueeze(1))
        self.value_function_optimizer.zero_grad()
        loss.backward()
        self.value_function_optimizer.step()

    def store_V(self, epidata):
        """
        Calculate and store standardized return overline_V for each step in a trajectory

        Args:
            epidata: A list of trajectory steps.

        Returns:
            A new list of steps with computed overline_V added to each step.
        """
        new_epidata = []
        for step in reversed(epidata):
            overline_V = step.reward + self.gamma * overline_V
            updated_step = step._replace(overline_V=overline_V)
            new_epidata.insert(0, updated_step)
        return new_epidata
