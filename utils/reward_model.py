import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import *
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NetworkBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self):
        super(NetworkBase, self).__init__()
    @abstractmethod
    def forward(self, x):
        return x

class Network(NetworkBase):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim,
                 activation_function = torch.relu, last_activation = None):
        """
        Full-connected layer network

        Args:
            layer_num: number of hidden layer
            input_dim: input feature dimension
            output_dim: output dimension
            hidden_dim: hidden layer size
            activation_function: activation function for hidden layers
            last_activation: activation after final layer, optimal
        """
        super(Network, self).__init__()
        self.activation = activation_function
        self.last_activation = last_activation
        
        # sizes of each layer
        layers_unit = [input_dim]+ [hidden_dim]*(layer_num-1) 
        
        # hidden layers
        layers = ([nn.Linear(layers_unit[idx],layers_unit[idx+1]) for idx in range(len(layers_unit)-1)])
        self.layers = nn.ModuleList(layers)
        
        # output layer
        self.last_layer = nn.Linear(layers_unit[-1],output_dim)
        
        # orthogonal initialization
        self.network_init()
        
    def forward(self, x):
        return self._forward(x)
    
    def _forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.last_layer(x)
        if self.last_activation != None:
            x = self.last_activation(x)
        return x
    
    def network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                layer.bias.data.zero_()


class Actor(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim,
                 activation_function = torch.tanh, last_activation = None,
                 trainable_std = False):
        super(Actor, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function ,last_activation)
        """
        Actor network for continuous control policies
        
        """
        self.trainable_std = trainable_std
        if self.trainable_std == True:
            self.logstd = nn.Parameter(torch.zeros(1, output_dim))
    def forward(self, x):
        mu = self._forward(x)
        if self.trainable_std == True:
            std = torch.exp(self.logstd)
        else:
            logstd = torch.zeros_like(mu)
            std = torch.exp(logstd)
        return mu,std

class Critic(Network):
    def __init__(self, layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation = None):
        """
        critic network for value estimation
        """
        super(Critic, self).__init__(layer_num, input_dim, output_dim, hidden_dim, activation_function, last_activation)
        
    def forward(self, *x):
        x = torch.cat(x,-1)
        return self._forward(x)


# ======================================================
# Encoder network for state to embedding
# ======================================================
class StateEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encode_dim, activation_function=torch.relu):
        super(StateEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, encode_dim)
        self.activation_function = activation_function

    def forward(self, state):
        x = self.activation_function(self.fc1(state))
        x = self.activation_function(self.fc2(x))
        state_embedding = self.activation_function(self.fc3(x))
        return state_embedding

# ======================================================
# Encoder network for action to embedding
# ======================================================
class ActionEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, encode_dim, activation_function=torch.relu):
        super(ActionEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, encode_dim)
        self.activation_function = activation_function

    def forward(self, action):
        x = self.activation_function(self.fc1(action))
        x = self.activation_function(self.fc2(x))
        action_embedding = self.activation_function(self.fc3(x))
        return action_embedding


# ======================================================
# Predicts reward from state and action embeddings
# ======================================================
class ForwardModel(nn.Module):
    def __init__(self, encode_dim, output_dim, last_activation=None):
        super(ForwardModel, self).__init__()
        self.last_fc = nn.Linear(encode_dim * 2, output_dim) 
        self.last_activation = last_activation

    def forward(self, state_embedding, action_embedding):
        x = torch.cat([state_embedding, action_embedding], dim=-1)  
        reward = self.last_fc(x)
        if self.last_activation is not None:
            reward = self.last_activation(reward)
        return reward

# ======================================================
# Full Reward Model: R(s, a), including a state encoder, action encoder, and forward model
# ======================================================
class Reward(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, encode_dim, output_dim=1, activation_function=torch.relu, last_activation=None):
        super(Reward, self).__init__()
        self.state_encoder = StateEncoder(state_dim, hidden_dim, encode_dim, activation_function)
        self.action_encoder = ActionEncoder(action_dim, hidden_dim, encode_dim, activation_function)
        self.forward_model = ForwardModel(encode_dim, output_dim, last_activation)

    def forward(self, state, action):
        state_embedding = self.state_encoder(state)
        action_embedding = self.action_encoder(action)
        reward = self.forward_model(state_embedding, action_embedding)
        return reward

    
