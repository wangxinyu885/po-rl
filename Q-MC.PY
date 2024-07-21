import Grid2OpOld
from Grid2OpOld.grid2op.Backend.pandaPowerBackend import PandaPowerBackend
from Grid2OpOld.grid2op.Reward.baseReward import BaseReward
from Grid2OpOld.grid2op.Reward.flatReward import FlatReward
from Grid2OpOld.grid2op.Environment.outage_env import OutageEnv
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from collections import deque, namedtuple
import gym
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)  # 增加神经元数量
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, sum(action_size))  # Output size is the sum of all action dimensions
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size, seed):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self):
        return len(self.memory)
class DQNAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.num_nodes = len(action_size)
        self.seed = random.seed(seed)
        
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=1e-4)
        
        self.memory = ReplayBuffer(buffer_size=int(1e5), batch_size=64, seed=seed)
        self.tau = 1e-3
        self.gamma = 0.99
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_every = 4
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
    
    def act(self, state, epsilon=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        if random.random() > epsilon:
            action_values = action_values.cpu().data.numpy().flatten()
            
            # 剔除已经故障的节点
            failed_nodes = np.where(state.cpu().numpy().flatten() == 0)[0]
            valid_nodes = np.setdiff1d(np.arange(self.num_nodes), failed_nodes)
            if len(valid_nodes) == 0:
                return (0, 0)  # 如果没有有效节点，返回一个默认动作
            
            node_index = valid_nodes[np.argmax(action_values[valid_nodes])]
            action_start = self.num_nodes + sum(self.action_size[:node_index])
            action_end = action_start + self.action_size[node_index]
            
            if action_start >= len(action_values):
                action_start = len(action_values) - 1
            if action_end > len(action_values):
                action_end = len(action_values)
            
            action_index = np.argmax(action_values[action_start:action_end])
            return (node_index, action_index)
        else:
            valid_nodes = np.setdiff1d(np.arange(self.num_nodes), np.where(state.cpu().numpy().flatten() == 0)[0])
            if len(valid_nodes) == 0:
                return (0, 0)  # 如果没有有效节点，返回一个默认动作
            node_index = random.choice(valid_nodes)
            action_index = random.choice(np.arange(self.action_size[node_index]))
            return (node_index, action_index)

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        node_indices = actions[:, 0]
        action_indices = actions[:, 1]
        
        Q_targets_next = self.qnetwork_target(next_states).detach()
        Q_targets_next_max = torch.gather(Q_targets_next[:, self.num_nodes:], 1, action_indices.view(-1, 1))
        
        Q_targets = rewards + (gamma * Q_targets_next_max * (1 - dones))
        
        Q_expected = torch.gather(self.qnetwork_local(states)[:, self.num_nodes:], 1, action_indices.view(-1, 1))
        
        loss = F.mse_loss(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)
    
    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

def train_dqn(env, agent, num_episodes, max_steps_per_episode=100):
    rewards = []
    for i_episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        while True:
            node_index, action_index = agent.act(state, agent.epsilon)
            gym_action = np.ones(len(env.action_space.nvec), dtype=int)
            gym_action[node_index] = action_index
            next_state, reward, done, _, _ = env.step(gym_action)
            agent.step(state, (node_index, action_index), reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1
            if done or step_count >= max_steps_per_episode:
                break
        rewards.append(total_reward)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon_decay * agent.epsilon)
        print(f"Episode {i_episode}/{num_episodes}, Total Reward: {total_reward}, Steps: {step_count}, Epsilon: {agent.epsilon:.2f}")
    return rewards
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_reward(x, y, title=""):
    """
    绘制奖励随迭代次数的变化图。
    """
    smoothed_y = smooth_curve(y)
    plt.figure()
    plt.plot(x, y, label="Original")
    plt.plot(x, smoothed_y, label="Smoothed")
    plt.title(title)
    plt.xlabel("迭代次数")
    plt.ylabel("奖励")
    plt.legend()
    plt.savefig('dqn_5000.png')
    plt.show()

env = OutageEnv()
state_size = env.observation_space.shape[0]
action_size = env.action_space.nvec  # 确保 action_size 是一个列表
agent = DQNAgent(state_size, action_size, seed=0)

rewards = train_dqn(env, agent, num_episodes=1000, max_steps_per_episode=100)
plot_reward(range(len(rewards)), rewards, "回合奖励变化")