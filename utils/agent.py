"""
for casestudy3

"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions.categorical as Categorical
import torch.nn.functional as F
from collections import deque
import copy

class Brain(nn.Module):
    def __init__(self, args):
        super(Brain, self).__init__()
        self.number_actions = args.number_actions
        self.fc1 = nn.Linear(3, 64)  
        self.fc2 = nn.Linear(64, 32)  
        self.fc3 = nn.Linear(32, args.number_actions)  
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)  

        self.optimizer = optim.Adam(self.parameters(), lr=args.learning_rate)

        self.loss_fn = nn.MSELoss()

    def forward(self, states):
        x = self.sigmoid(self.fc1(states))  
        x = self.sigmoid(self.fc2(x))       
        q_values = self.softmax(self.fc3(x))  
        return q_values

    def train_step(self, states, target_q_values):
        
        predictions = self.forward(states)
        loss = self.loss_fn(predictions, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class DQN(object):
    def __init__(self, args):
        self.memory = list()
        self.max_memory = args.ax_memory
        self.discount = args.discount

    def remember(self, transition, game_over):
        self.memory.append([transition, game_over])
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size = 10):
        len_memory = len(self.memory)
        num_inputs = self.memory[0][0][0].shape[1]
        
        num_outputs = model.number_actions
        inputs = np.zeros((min(len_memory, batch_size), num_inputs))
        targets = np.zeros((min(len_memory, batch_size), num_outputs))
        for i, idx in enumerate(np.random.randint(0, len_memory, size = min(len_memory, batch_size))):
            current_state, action, reward, next_state = self.memory[idx][0]
            game_over = self.memory[idx][1]
            current_state_tensor = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                current_q_values = model(current_state_tensor).squeeze(0).numpy()
                next_q_values = model(next_state_tensor).squeeze(0).numpy()
            inputs[i] = current_state
            targets[i] = current_q_values

            Q_sa = np.max(next_q_values)

            if game_over:
                targets[i, action] = reward
            else:
                targets[i, action] = reward + self.discount * Q_sa
        return inputs, targets


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, net_width):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, net_width)
        self.l2 = nn.Linear(net_width, net_width)
        self.l3 = nn.Linear(net_width, action_dim)

    def forward(self, state):
        n = torch.tanh(self.l1(state))
        n = torch.tanh(self.l2(n))
        return n

    def pi(self, state, softmax_dim = -1):
        n = self.forward(state)
        n = self.l3(n)
        prob = F.softmax(n, dim=softmax_dim)
        return prob

class Critic(nn.Module):
    def __init__(self, state_dim,net_width):
        super(Critic, self).__init__()

        self.C1 = nn.Linear(state_dim, net_width)
        self.C2 = nn.Linear(net_width, net_width)
        self.C3 = nn.Linear(net_width, 1)

    def forward(self, state):
        v = torch.sigmoid(self.C1(state))
        v = torch.sigmoid(self.C2(v))
        v = self.C3(v)
        return v


class PPO_agent():
    def __init__(self, args, device):
        self.actor = Actor(args.state_dim, args.action_dim, args.net_width).to(device)
        self.critic = Critic(args.state_dim, args.net_width).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.lr)

        self.gamma = args.gamma
        self.lambd = args.lambd
        self.clip_rate = args.clip_rate
        self.K_epochs = args.K_epochs
        self.batch_size = args.batch_size
        self.entropy_coef = args.entropy_coef
        self.device = device

        
        self.max_memory = args.max_memory
        self.memory = deque(maxlen=self.max_memory)

    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            probs = self.actor.pi(state)
        if deterministic:
            action = torch.argmax(probs).item()
        else:
            dist = Categorical(probs)
            action = dist.sample().item()
        logprobs = torch.log(probs.squeeze() + 1e-8)  
        logprobs = logprobs.detach().cpu().numpy()  
        return action, logprobs[action], logprobs
    
    def store_transition(self, state, action, reward, next_state, logprob, done):
        self.memory.append((state, action, reward, next_state, logprob, done))

    def train(self):
        states, actions, rewards, next_states, logprobs, dones = zip(*self.memory)
        states = torch.tensor(states, dtype=torch.float32).squeeze(1).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).squeeze(1).to(self.device)
        logprobs = torch.tensor(logprobs).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            values = self.critic(states).squeeze()
            next_values = self.critic(next_states).squeeze()
            td_target = rewards + self.gamma * next_values * (~dones)
            deltas = td_target - values
            advantages = []
            advantage = 0
            for delta, done in zip(reversed(deltas), reversed(dones)):
                advantage = delta + (self.gamma * self.lambd * advantage * (~done))
                advantages.insert(0, advantage)
            advantages = torch.tensor(advantages, dtype=torch.float32).to(self.device)

        for _ in range(self.K_epochs):
            
            for _ in range(len(states) // self.batch_size):
                slice_idx = torch.randperm(len(states))[:self.batch_size]
                
                batch_states = states[slice_idx]
                batch_actions = actions[slice_idx]
                batch_advantages = advantages[slice_idx]
                batch_logprobs = logprobs[slice_idx]
                batch_td_target = td_target[slice_idx]

                
                new_probs = self.actor.pi(batch_states) 
                new_probs = new_probs.gather(1, batch_actions.unsqueeze(1)) 
                new_probs = new_probs.squeeze() 
                entropy = Categorical(self.actor.pi(batch_states)).entropy().mean()
                ratios = torch.exp(torch.log(new_probs) - batch_logprobs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_rate, 1 + self.clip_rate) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

                
                critic_loss = ((self.critic(batch_states).squeeze() - batch_td_target) ** 2).mean()

                
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()


    def save(self):
        torch.save(self.critic.state_dict(), "./model/ppo+_critic.pth")
        torch.save(self.actor.state_dict(), "./model/ppo+_actor.pth")

class Double_Q_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Double_Q_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        
        self.Q1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q2 = build_net(layers, nn.ReLU, nn.Identity)
    
    def forward(self, s):
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        return q1,q2

class Policy_Net(nn.Module):
    def __init__(self, state_dim, action_dim, hid_shape):
        super(Policy_Net, self).__init__()
        layers = [state_dim] + list(hid_shape) + [action_dim]
        self.P = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, s):
        logits = self.P(s)
        probs = F.softmax(logits, dim=1)
        return probs

class SACD_agent():
    def __init__(self, state_dim, action_dim, hid_shape, gamma, alpha, batch_size, adaptive_alpha, lr, dvc):
        self.state_dim=state_dim
        self.action_dim=action_dim
        self.hid_shape=hid_shape
        self.gamma=gamma
        self.alpha=alpha
        self.batch_size=batch_size
        self.adaptive_alpha=adaptive_alpha
        self.lr=lr
        self.dvc=dvc
        self.tau = 0.005
        self.H_mean = 0
        self.replay_buffer = ReplayBuffer(self.state_dim, self.dvc, max_size=int(1e6))

        self.actor = Policy_Net(self.state_dim, self.action_dim, self.hid_shape).to(self.dvc)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.q_critic = Double_Q_Net(self.state_dim, self.action_dim, self.hid_shape).to(self.dvc)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        for p in self.q_critic_target.parameters(): p.requires_grad = False

        
        if self.adaptive_alpha:
            self.target_entropy = 0.6 * (-np.log(1 / self.action_dim))  
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def select_action(self, state, deterministic=False):
        state = torch.tensor(state, dtype=torch.float32).to(self.dvc)
        
        with torch.no_grad():
            
            probs = self.actor(state)
            if deterministic:
                action = probs.argmax(dim=-1).item()
            else:
                action = Categorical(probs).sample().item()
            logprobs = torch.log(probs.squeeze() + 1e-8)  
            logprobs = logprobs.detach().cpu().numpy()  
            
            return action, logprobs[action], logprobs
    
    def train(self):
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            next_probs = self.actor(next_states)  
            next_log_probs = torch.log(next_probs + 1e-8)  
            next_q1_all, next_q2_all = self.q_critic_target(next_states)  
            min_next_q_all = torch.min(next_q1_all, next_q2_all)
            v_next = torch.sum(next_probs * (min_next_q_all - self.alpha * next_log_probs), dim=1, keepdim=True)  
            target_Q = rewards + (~dones) * self.gamma * v_next

        q1_all, q2_all = self.q_critic(states)  
        q1 = q1_all.gather(1, actions.long())  
        q2 = q2_all.gather(1, actions.long())
        q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)

        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()
        
        probs = self.actor(states)  
        log_probs = torch.log(probs + 1e-8)
        with torch.no_grad():
            q1_all, q2_all = self.q_critic(states)
        min_q_all = torch.min(q1_all, q2_all)

        actor_loss = torch.sum(probs * (self.alpha * log_probs - min_q_all), dim=1).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.adaptive_alpha:
            with torch.no_grad():
                self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
            alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()

        
        for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, teacher):
        torch.save(self.actor.state_dict(), f"./model/SAC+_actor.pth")
        torch.save(self.q_critic.state_dict(), f"./model/SAC+_critic.pth")

class ReplayBuffer(object):
    def __init__(self, state_dim, dvc, max_size=int(1e6)):
        self.max_size = max_size
        self.dvc = dvc
        self.ptr = 0
        self.size = 0

        self.s = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.dvc)
        self.a = torch.zeros((max_size, 1),dtype=torch.long,device=self.dvc)
        self.r = torch.zeros((max_size, 1),dtype=torch.float,device=self.dvc)
        self.s_next = torch.zeros((max_size, state_dim),dtype=torch.float,device=self.dvc)
        self.dw = torch.zeros((max_size, 1),dtype=torch.bool,device=self.dvc)

    def add(self, s, a, r, s_next, dw):
        self.s[self.ptr] = torch.from_numpy(s).to(self.dvc)
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s_next[self.ptr] = torch.from_numpy(s_next).to(self.dvc)
        self.dw[self.ptr] = dw

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = torch.randint(0, self.size, device=self.dvc, size=(batch_size,))
        return self.s[ind], self.a[ind], self.r[ind], self.s_next[ind], self.dw[ind]
     

def build_net(layer_shape, hid_activation, output_activation):
    
    layers = []
    for j in range(len(layer_shape)-1):
        act = hid_activation if j < len(layer_shape)-2 else output_activation
        layers += [nn.Linear(layer_shape[j], layer_shape[j+1]), act()]
    return nn.Sequential(*layers)
