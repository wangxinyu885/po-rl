import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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
import torch.nn.functional as F
from Grid2OpOld.grid2op.gym_compat import DiscreteActSpace
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# class Actor(nn.Module):
#     def __init__(self, state_size, action_size, seed):
#         super(Actor, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, action_size)
        
#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return F.softmax(self.fc3(x), dim=-1)

# class Critic(nn.Module):
#     def __init__(self, state_size, seed):
#         super(Critic, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         self.fc1 = nn.Linear(state_size, 128)
#         self.fc2 = nn.Linear(128, 128)
#         self.fc3 = nn.Linear(128, 1)
        
#     def forward(self, state):
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         return self.fc3(x)

# class ReplayBuffer:
#     def __init__(self, buffer_size, batch_size, seed):
#         self.memory = deque(maxlen=buffer_size)
#         self.batch_size = batch_size
#         self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
#         self.seed = random.seed(seed)
    
#     def add(self, state, action, reward, next_state, done):
#         e = self.experience(state, action, reward, next_state, done)
#         self.memory.append(e)
    
#     def sample(self):
#         experiences = random.sample(self.memory, k=self.batch_size)
        
#         states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
#         actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
#         rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
#         next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
#         dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
#         return (states, actions, rewards, next_states, dones)
    
#     def __len__(self):
#         return len(self.memory)

# class ACAgent:
#     def __init__(self, state_size, action_size, seed):
#         self.state_size = state_size
#         self.action_size = action_size
#         self.seed = random.seed(seed)
        
#         self.actor = Actor(state_size, action_size, seed).to(device)
#         self.critic = Critic(state_size, seed).to(device)
#         self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
#         self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)
        
#         self.memory = ReplayBuffer(buffer_size=int(1e5), batch_size=64, seed=seed)
#         self.gamma = 0.99
        
#         self.epsilon = 1.0
#         self.epsilon_decay = 0.995
#         self.epsilon_min = 0.01
#         self.update_every = 4
#         self.t_step = 0
        
#     def step(self, state, action, reward, next_state, done):
#         self.memory.add(state, action, reward, next_state, done)
        
#         self.t_step = (self.t_step + 1) % self.update_every
#         if self.t_step == 0:
#             if len(self.memory) > self.memory.batch_size:
#                 experiences = self.memory.sample()
#                 self.learn(experiences, self.gamma)
    
#     def act(self, state, epsilon=0.0):
#         state = torch.from_numpy(state).float().unsqueeze(0).to(device)
#         self.actor.eval()
#         with torch.no_grad():
#             action_probs = self.actor(state)
#         self.actor.train()
        
#         if random.random() > epsilon:
#             action = np.argmax(action_probs.cpu().data.numpy())
#         else:
#             action = random.choice(np.arange(self.action_size))
#         return action

#     def learn(self, experiences, gamma):
#         states, actions, rewards, next_states, dones = experiences
        
#         # 更新Critic
#         Q_targets_next = self.critic(next_states).detach()
#         Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
#         Q_expected = self.critic(states)
#         critic_loss = F.mse_loss(Q_expected, Q_targets)
        
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()
        
#         # 更新Actor
#         action_probs = self.actor(states)
#         actions_one_hot = torch.zeros_like(action_probs).scatter_(1, actions, 1)
#         actor_loss = -torch.mean(torch.sum(actions_one_hot * torch.log(action_probs), dim=1) * (Q_targets - Q_expected.detach()))
        
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()
        
#         return actor_loss.item(), critic_loss.item()

# def train_ac(env, agent, num_episodes, max_steps_per_episode=100):
#     rewards = []
#     actor_losses = []
#     critic_losses = []
#     for i_episode in range(1, num_episodes + 1):
#         state, _ = env.reset()
#         total_reward = 0
#         step_count = 0
#         while True:
#             action = agent.act(state, agent.epsilon)
#             next_state, reward, done, _, _ = env.step(action)
#             agent.step(state, action, reward, next_state, done)
#             state = next_state
#             total_reward += reward
#             step_count += 1
#             if done or step_count >= max_steps_per_episode:
#                 break
#         rewards.append(total_reward)
#         agent.epsilon = max(agent.epsilon_min, agent.epsilon_decay * agent.epsilon)

#         if len(agent.memory) > agent.memory.batch_size:
#             # 打印并记录损失
#             actor_loss, critic_loss = agent.learn(agent.memory.sample(), agent.gamma)
#             actor_losses.append(actor_loss)
#             critic_losses.append(critic_loss)
#         else:
#             actor_losses.append(None)
#             critic_losses.append(None)

#         print(f"Episode {i_episode}/{num_episodes}, Total Reward: {total_reward}, Steps: {step_count}, "
#               f"Epsilon: {agent.epsilon:.2f}, Actor Loss: {actor_losses[-1]}, Critic Loss: {critic_losses[-1]}")

#     return rewards, actor_losses, critic_losses

# def smooth_curve(points, factor=0.9):
#     smoothed_points = []
#     for point in points:
#         if point is None:
#             continue
#         if smoothed_points:
#             previous = smoothed_points[-1]
#             smoothed_points.append(previous * factor + point * (1 - factor))
#         else:
#             smoothed_points.append(point)
#     return smoothed_points

# def plot_metrics(x, rewards, actor_losses, critic_losses):
#     smoothed_rewards = smooth_curve(rewards)
#     smoothed_actor_losses = smooth_curve([l for l in actor_losses if l is not None])
#     smoothed_critic_losses = smooth_curve([l for l in critic_losses if l is not None])
    
#     plt.figure()
#     plt.subplot(3, 1, 1)
#     plt.plot(x, rewards, label="Original")
#     plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label="Smoothed")
#     plt.title("Episode Rewards")
#     plt.xlabel("Episode")
#     plt.ylabel("Reward")
#     plt.legend()
    
#     plt.subplot(3, 1, 2)
#     plt.plot(x, [l if l is not None else 0 for l in actor_losses], label="Actor Loss")
#     plt.plot(range(len(smoothed_actor_losses)), smoothed_actor_losses, label="Smoothed Actor Loss")
#     plt.title("Actor Loss")
#     plt.xlabel("Episode")
#     plt.ylabel("Loss")
#     plt.legend()

#     plt.subplot(3, 1, 3)
#     plt.plot(x, [l if l is not None else 0 for l in critic_losses], label="Critic Loss")
#     plt.plot(range(len(smoothed_critic_losses)), smoothed_critic_losses, label="Smoothed Critic Loss")
#     plt.title("Critic Loss")
#     plt.xlabel("Episode")
#     plt.ylabel("Loss")
#     plt.legend()

#     plt.tight_layout()
#     plt.savefig('ac_metrics1.png')
#     plt.show()

# env = OutageEnv()
# state_size = env.observation_space.shape[0]
# action_size = env.action_space.nvec.sum()
# agent = ACAgent(state_size, action_size, seed=0)

# rewards, actor_losses, critic_losses = train_ac(env, agent, num_episodes=1000, max_steps_per_episode=100)
# plot_metrics(range(len(rewards)), rewards, actor_losses, critic_losses)

class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_size, seed):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        
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

class ACAgent:
    def __init__(self, state_size, action_size, seed, env):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.env = env

        self.actor = Actor(state_size, action_size, seed).to(device)
        self.critic = Critic(state_size, seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-4)

        self.memory = ReplayBuffer(buffer_size=int(1e5), batch_size=64, seed=seed)
        self.gamma = 0.99

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.update_every = 4
        self.t_step = 0

        self.initial_action_probs = None
        self.final_action_probs = None
        self.action_prob_history = []  # 用于记录动作概率的历史

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)
    
    def act(self, state, epsilon=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action_probs = self.actor(state)
        self.actor.train()

        action_probs = action_probs.cpu().data.numpy().flatten()

        # 记录初始和最终动作概率
        if self.initial_action_probs is None:
            self.initial_action_probs = action_probs.copy()
        self.final_action_probs = action_probs.copy()

        # 记录动作概率
        self.action_prob_history.append(action_probs)

        if random.random() > epsilon:
            action = np.argmax(action_probs)
        else:
            action = random.choice(np.arange(self.action_size))
        return action
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        # 更新Critic
        Q_targets_next = self.critic(next_states).detach()
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic(states)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        action_probs = self.actor(states)
        actions_one_hot = torch.zeros_like(action_probs).scatter_(1, actions, 1)
        actor_loss = -torch.mean(torch.sum(actions_one_hot * torch.log(action_probs), dim=1) * (Q_targets - Q_expected.detach()))
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        return actor_loss.item(), critic_loss.item()

def train_ac(env, agent, num_episodes, max_steps_per_episode=100):
    rewards = []
    actor_losses = []
    critic_losses = []
    for i_episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        while True:
            action = agent.act(state, agent.epsilon)
            next_state, reward, done, _, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step_count += 1
            if done or step_count >= max_steps_per_episode:
                break
        rewards.append(total_reward)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon_decay * agent.epsilon)

        if len(agent.memory) > agent.memory.batch_size:
            # 打印并记录损失
            actor_loss, critic_loss = agent.learn(agent.memory.sample(), agent.gamma)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
        else:
            actor_losses.append(None)
            critic_losses.append(None)

        print(f"Episode {i_episode}/{num_episodes}, Total Reward: {total_reward}, Steps: {step_count}, "
              f"Epsilon: {agent.epsilon:.2f}, Actor Loss: {actor_losses[-1]}, Critic Loss: {critic_losses[-1]}")

    return rewards, actor_losses, critic_losses, agent.action_prob_history


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if point is None:
            continue
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_metrics(x, rewards, actor_losses, critic_losses):
    # 绘制奖励、Actor Loss和Critic Loss
    plt.figure()
    
    plt.subplot(3, 1, 1)
    smoothed_rewards = smooth_curve(rewards)
    plt.plot(x, rewards, label="Original")
    plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label="Smoothed")
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    
    plt.subplot(3, 1, 2)
    smoothed_actor_losses = smooth_curve([l for l in actor_losses if l is not None])
    plt.plot(x, [l if l is not None else 0 for l in actor_losses], label="Actor Loss")
    plt.plot(range(len(smoothed_actor_losses)), smoothed_actor_losses, label="Smoothed Actor Loss")
    plt.title("Actor Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(3, 1, 3)
    smoothed_critic_losses = smooth_curve([l for l in critic_losses if l is not None])
    plt.plot(x, [l if l is not None else 0 for l in critic_losses], label="Critic Loss")
    plt.plot(range(len(smoothed_critic_losses)), smoothed_critic_losses, label="Smoothed Critic Loss")
    plt.title("Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig('ac_metrics.png')
    plt.show()

def plot_action_prob_comparison(initial_probs, final_probs):
    actions = np.arange(len(initial_probs))
    width = 0.35  # 设置柱状图的宽度

    plt.figure()
    plt.bar(actions - width/2, initial_probs, width, label='Initial Probabilities')
    plt.bar(actions + width/2, final_probs, width, label='Final Probabilities')

    plt.xlabel('Action')
    plt.ylabel('Probability')
    plt.title('Comparison of Initial and Final Action Probabilities')
    plt.xticks(actions)
    plt.legend()

    plt.tight_layout()
    plt.savefig('action_prob_comparison.png')
    plt.show()

env = OutageEnv()
state_size = env.observation_space.shape[0]
# 修正动作空间大小的获取方式
action_size = env.action_space.n

agent = ACAgent(state_size, action_size, seed=0, env=env)
# 训练并获取指标数据
rewards, actor_losses, critic_losses, action_prob_history = train_ac(env, agent, num_episodes=1000, max_steps_per_episode=100)

# 绘制指标图
plot_metrics(range(len(rewards)), rewards, actor_losses, critic_losses)

# 绘制初始和最终动作概率对比图
plot_action_prob_comparison(agent.initial_action_probs, agent.final_action_probs)

