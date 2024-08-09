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
import collections


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
        # 将 OrderedDict 的值组合为一个 NumPy 数组
        if isinstance(state, collections.OrderedDict):
            state = np.concatenate([np.atleast_1d(v) for v in state.values()])
        print("State shape:", state.shape)  # 打印状态的形状
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
    # def learn(self, experiences, gamma):
    #     states, actions, rewards, next_states, dones = experiences
        
    #     # Critic 更新
    #     Q_targets_next = self.critic(next_states).detach()
    #     Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
    #     Q_expected = self.critic(states)
    #     critic_loss = F.mse_loss(Q_expected, Q_targets)
        
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optimizer.step()
        
    #     # Actor 更新
    #     action_probs = self.actor(states)
    #     actions_one_hot = torch.zeros_like(action_probs).scatter_(1, actions, 1)
    #     actor_loss = -torch.mean(torch.sum(actions_one_hot * torch.log(action_probs), dim=1) * (Q_targets - Q_expected.detach()))
        
    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()
    #     self.actor_optimizer.step()
        
    #     # 打印负载信息
    #     load_pos = self.env.load_p_pos()  # 获取负载信息的起始位置
    #     load_info = states[:, load_pos:load_pos + self.env._env.n_load].cpu().data.numpy()  # 获取负载信息
    #     total_load = np.sum(load_info, axis=1)
    #     # print("Load Info:", load_info)
    #     # print("Total Load:", total_load)
        
    #     return actor_loss.item(), critic_loss.item()
    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences
        
        # 获取负载信息并计算负载奖励
        load_pos = self.env.load_p_pos()  # 获取负载信息的起始位置
        load_info = states[:, load_pos:load_pos + self.env._env.n_load]  # 获取负载信息
        total_load = torch.sum(load_info, dim=1)  # 计算总负载
        load_reward = total_load * 0.5  # 假设负载奖励的权重为0.5

        # Critic 更新
        Q_targets_next = self.critic(next_states).detach()
        Q_targets = rewards + load_reward.unsqueeze(1) + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic(states)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor 更新
        action_probs = self.actor(states)
        actions_one_hot = torch.zeros_like(action_probs).scatter_(1, actions, 1)
        actor_loss = -torch.mean(torch.sum(actions_one_hot * torch.log(action_probs), dim=1) * (Q_targets - Q_expected.detach()))
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 打印负载信息
        load_info = states[:, load_pos:load_pos + self.env._env.n_load].cpu().data.numpy()  # 获取负载信息
        total_load = np.sum(load_info, axis=1)
        print("Load Info:", load_info)
        print("Total Load:", total_load)
        
        return actor_loss.item(), critic_loss.item()


def train_ac(env, agent, num_episodes, max_steps_per_episode=100):
    rewards = []
    actor_losses = []
    critic_losses = []
    for i_episode in range(1, num_episodes + 1):
        obs, reward, done, truncated, info = env.reset()
        total_reward = 0
        step_count = 0
        while not done and step_count < max_steps_per_episode:
            action = agent.act(obs, agent.epsilon)
            next_obs, reward, done, truncated, info = env.step(action)
            agent.step(obs, action, reward, next_obs, done)
            obs = next_obs
            total_reward += reward
            step_count += 1
        rewards.append(total_reward)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon_decay * agent.epsilon)

        if len(agent.memory) > agent.memory.batch_size:
            actor_loss, critic_loss = agent.learn(agent.memory.sample(), agent.gamma)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
        else:
            actor_losses.append(None)
            critic_losses.append(None)
        # 打印负载信息
        if env.g2op_obs is not None:
            load_info = env.g2op_obs.load_p
            total_load = np.sum(load_info)
            print(f"Episode {i_episode}/{num_episodes}, Total Load: {total_load}, Load Info: {load_info}")
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
    # 绘制奖励图
    plt.figure()
    smoothed_rewards = smooth_curve(rewards)
    plt.plot(x, rewards, label="Original")
    plt.plot(range(len(smoothed_rewards)), smoothed_rewards, label="Smoothed")
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.ylim(-10, 0)  # 限制 y 轴范围在 -10 到 -1
    plt.legend()
    plt.savefig('episode_rewards2.png')
    plt.show()

    # 绘制 Actor Loss 图
    plt.figure()
    smoothed_actor_losses = smooth_curve([l for l in actor_losses if l is not None])
    plt.plot(x, [l if l is not None else 0 for l in actor_losses], label="Actor Loss")
    plt.plot(range(len(smoothed_actor_losses)), smoothed_actor_losses, label="Smoothed Actor Loss")
    plt.title("Actor Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('actor_loss2.png')
    plt.show()

    # 绘制 Critic Loss 图
    plt.figure()
    smoothed_critic_losses = smooth_curve([l for l in critic_losses if l is not None])
    plt.plot(x, [l if l is not None else 0 for l in critic_losses], label="Critic Loss")
    plt.plot(range(len(smoothed_critic_losses)), smoothed_critic_losses, label="Smoothed Critic Loss")
    plt.title("Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('critic_loss2.png')
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
    plt.savefig('action_prob_comparison2.png')
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

