import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import Grid2OpOld
from Grid2OpOld.grid2op.Backend.pandaPowerBackend import PandaPowerBackend
from Grid2OpOld.grid2op.Reward.baseReward import BaseReward
from Grid2OpOld.grid2op.Reward.flatReward import FlatReward
import gym
from Grid2OpOld.grid2op.Environment.outage_env import OutageEnv
from collections import deque
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.priorities = []
        self.pos = 0
        self.alpha = alpha

    def push(self, error, transition):
        # 确保优先级是实数
        priority = abs(float(np.real((error + 1e-5) ** self.alpha)))
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        total_priority = sum(self.priorities)
        # 确保概率是实数
        probs = [float(np.real(p / total_priority)) for p in self.priorities]
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        transitions = [self.buffer[idx] for idx in indices]
        weights = [(len(self.buffer) * probs[idx]) ** (-beta) for idx in indices]
        max_weight = max(weights)
        weights = [float(np.real(w / max_weight)) for w in weights]
        return transitions, weights, indices

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            # 确保优先级是实数
            self.priorities[idx] = abs(float(np.real((error + 1e-5) ** self.alpha)))


def print_action_space(env):
    """
    打印每个状态或节点可以执行的动作。
    
    参数:
        env: 环境对象
    """
    print("Action space details:")
    action_space = env.action_space
    print(action_space)  # 打印动作空间的详细信息
    try:
        nvec = action_space.nvec
        for i, n in enumerate(nvec):
            print(f"Action space for node {i}: {n} possible actions")
    except AttributeError:
        print("Action space does not have 'nvec' attribute.")
def create_random_policy(env):
    def policy_fn(state):
        nA = len(env.action_space.nvec)
        action = np.zeros(nA, dtype=int)
        action_probs = np.zeros(nA)

        if len(state) < nA:
            state = np.pad(state, (0, nA - len(state)), 'constant')
        elif len(state) > nA:
            state = state[:nA]

        for i in range(nA):
            valid_actions = [a for a in range(env.action_space.nvec[i]) if state[i] != 0]
            if len(valid_actions) == 0:
                action[i] = 0
                action_probs[i] = 1.0
            else:
                action[i] = np.random.choice(valid_actions)
                action_probs[i] = 1.0 / len(valid_actions)
        
        return action, action_probs
    
    return policy_fn

def create_epsilon_greedy_policy(Q, epsilon, action_space):
    def policy_fn(state):
        state_ = tuple(state)
        nA = len(action_space.nvec)
        action = np.zeros(nA, dtype=int)

        if len(state) < nA:
            state = np.pad(state, (0, nA - len(state)), 'constant')
        elif len(state) > nA:
            state = state[:nA]

        for i in range(nA):
            if random.random() > epsilon:
                if state_ in Q and Q[state_]:
                    valid_actions = [a for a in range(action_space.nvec[i]) if state[i] != 0]
                    if len(valid_actions) == 0:
                        action[i] = 0
                    else:
                        best_action = max(valid_actions, key=lambda x: Q[state_][(i, x)])
                        action[i] = best_action
                else:
                    action[i] = np.random.choice(action_space.nvec[i])
            else:
                valid_actions = [a for a in range(action_space.nvec[i]) if state[i] != 0]
                if len(valid_actions) == 0:
                    action[i] = 0
                else:
                    action[i] = np.random.choice(valid_actions)
                
        return action
    
    return policy_fn

# def create_random_policy(env):
#     """
#     返回随机动作和动作概率。
    
#     参数:
#         env: 环境对象
    
#     返回:
#         一个随机动作和对应的动作概率
#     """
#     nA = len(env.action_space)
#     action = np.zeros(nA, dtype=int)
#     action_probs = np.zeros(nA)
    
#     for i in range(nA):
#         action[i] = np.random.choice(env.action_space[i].n)
#         action_probs[i] = 1.0 / env.action_space[i].n
    
#     return action, action_probs

def create_greedy_policy(Q, action_space):
    """
    创建一个基于Q值的贪婪策略。
    
    参数:
        Q: 从状态到动作值的映射字典
        action_space: 动作空间
    
    返回:
        一个函数，输入观测值，返回动作的概率向量
    """
    def policy_fn(state):
        state_ = tuple(state)
        action_values = Q[state_]
        best_action = max(action_values.keys(), key=lambda x: action_values[x])
        return best_action
    return policy_fn

# def create_epsilon_greedy_policy(Q, epsilon, action_space):
#     def policy_fn(state):
#         state_ = tuple(state)
#         action = np.zeros(len(action_space), dtype=int)
        
#         for i in range(len(action_space)):
#             if random.random() > epsilon:
#                 if state_ in Q and Q[state_]:
#                     best_action = max(Q[state_].keys(), key=lambda x: Q[state_][x])[i]
#                 else:
#                     best_action = np.random.choice(action_space[i].n)
#                 action[i] = best_action
#             else:
#                 action[i] = np.random.choice(action_space[i].n)
                
#         return action
    
#     return policy_fn

def mc_control_importance_sampling(env, num_episodes, behavior_policy, discount_factor=0.9, epsilon=0.1):
    """
    使用加权重要性采样的蒙特卡洛控制算法进行离策略控制。
    找到一个最优的贪婪策略。
    
    参数:
        env: OpenAI gym环境
        num_episodes: 采样的回合数
        behavior_policy: 生成回合时遵循的行为策略
        discount_factor: 折扣因子Gamma
        epsilon: 探索概率
    
    返回:
        一个元组(Q, policy)
        Q是将状态映射到动作值的字典
        policy是一个函数，输入观测值，返回动作的概率。这是最优的贪婪策略
    """
    def nested_defaultdict():
        return defaultdict(float)
    
    Q = defaultdict(nested_defaultdict)
    C = defaultdict(nested_defaultdict)
    target_policy = create_epsilon_greedy_policy(Q, epsilon, env.action_space.nvec.size)
    total_reward = []

    for i_episode in range(1, num_episodes + 1):
        print(f"\nEpisode {i_episode}/{num_episodes}")
        
        episode = []
        obs, info = env.reset()
        if obs is None:
            continue
        else:
            state = tuple(obs)
            episode_reward = 0
        
        for t in range(100):
            action, _ = behavior_policy(env)
            
            # 逐步探索：一次只选择一个动作进行尝试
            individual_actions = np.eye(env.action_space.nvec.size, dtype=int)
            for sub_action in individual_actions:
                next_state, reward, done, _, _ = env.step(sub_action)
                episode_reward += reward
                episode.append((state, sub_action, reward))
                print(f"Step {t}, Sub-action: {sub_action}, Reward: {reward}, Next state: {next_state}, Done: {done}")
                if done:
                    break
                state = tuple(next_state)
            
            print(f"Step {t}, Action: {action}, Reward: {reward}")
            if done:
                print(f"Episode ended after {t+1} steps")
                break

        total_reward.append(episode_reward)
        G = 0.0
        W = 1.0
        
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            state_ = tuple(state)
            action_ = tuple(action)
            G = discount_factor * G + reward
            C[state_][action_] += W
            Q[state_][action_] += (W / C[state_][action_]) * (G - Q[state_][action_])
            # print(f"State: {state_}, Action: {action_}, G: {G}, W: {W}, Q[{state_}][{action_}]: {Q[state_][action_]}")
            if not np.array_equal(action_, target_policy(state)):
                break
            _, action_probs = behavior_policy(env)
            W = W * 1.0 / action_probs

    return Q, target_policy, total_reward


# 软更新，多步回报
def soft_update(target, source, tau):
    for target_param, param in zip(target.values(), source.values()):
        target_param += tau * (param - target_param)

# def mc_control_importance_sampling_hierarchical(env, num_episodes, behavior_policy, discount_factor=1, epsilon=0.1, max_steps_per_episode=100, alpha=0.6, beta_start=0.4, beta_end=1.0, n_step=3, tau=0.01):
#     def nested_defaultdict():
#         return defaultdict(float)
    
#     Q = defaultdict(nested_defaultdict)
#     target_Q = defaultdict(nested_defaultdict)
#     C = defaultdict(nested_defaultdict)
#     epsilon_start = 1.0
#     epsilon_end = 0.1
#     epsilon_decay = 0.995
#     total_reward = []
#     buffer = PrioritizedReplayBuffer(capacity=10000, alpha=alpha)

#     for i_episode in range(1, num_episodes + 1):
#         epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** i_episode))
#         beta = beta_start + (beta_end - beta_start) * (i_episode / num_episodes)
#         target_policy = create_epsilon_greedy_policy(Q, epsilon, env.action_space)
        
#         print(f"\nEpisode {i_episode}/{num_episodes}")
        
#         episode = []
#         obs, info = env.reset()
#         if obs is None:
#             continue
#         else:
#             state = tuple(obs)
#             episode_reward = 0
#             print(f"Initial state: {state}")
        
#         for t in range(max_steps_per_episode):
#             node = np.random.choice(len(env.action_space))
#             sub_action = np.zeros(len(env.action_space), dtype=int)
#             sub_action[node] = np.random.choice(env.action_space[node].n)
#             next_state, reward, done, _, _ = env.step(sub_action)
#             episode_reward += reward
#             episode.append((state, sub_action, reward))
#             print(f"Step {t}, Node: {node}, Sub-action: {sub_action}, Reward: {reward}, Next state: {next_state}, Done: {done}")
            
#             state = tuple(next_state)
#             print(f"Step {t}, Reward: {reward}")

#             if done:
#                 print(f"Episode ended after {t+1} steps")
#                 break

#         total_reward.append(episode_reward)

#         # 多步回报
#         G = 0.0
#         W = 1.0
#         for t in range(len(episode))[::-1]:
#             state, action, reward = episode[t]
#             state_ = tuple(state)
#             action_ = tuple(action)
#             G = discount_factor * G + reward
#             C[state_][action_] += W
#             buffer.push(G, (state, action, reward))
#             if len(buffer.buffer) > 32:
#                 mini_batch, weights, indices = buffer.sample(32, beta)
#                 errors = []
#                 for (s, a, r), w in zip(mini_batch, weights):
#                     s_ = tuple(s)
#                     a_ = tuple(a)
#                     Q[s_][a_] += w * (G - Q[s_][a_])
#                     errors.append(abs(G - Q[s_][a_]))
#                 buffer.update_priorities(indices, errors)
#             if not np.array_equal(action_, target_policy(state)):
#                 break
#             _, action_probs = behavior_policy(env)
#             W = W * 1.0 / action_probs

#         # 软更新目标Q值
#         soft_update(target_Q, Q, tau)
#         # 打印当前最优策略
#         if i_episode % 100 == 0 or i_episode == num_episodes:
#             print("\nCurrent best policy at episode {}:".format(i_episode))
#             for state in Q.keys():
#                 best_action = max(Q[state].keys(), key=lambda x: Q[state][x])
#                 print("State: {}, Best action: {}, Q-value: {}".format(state, best_action, Q[state][best_action]))

#     return Q, target_policy, total_reward

def mc_control_importance_sampling_hierarchical(env, num_episodes, behavior_policy, discount_factor=1, epsilon=0.1, max_steps_per_episode=100, alpha=0.6, beta_start=0.4, beta_end=1.0, n_step=3, tau=0.01):
    def nested_defaultdict():
        return defaultdict(float)
    
    Q = defaultdict(nested_defaultdict)
    target_Q = defaultdict(nested_defaultdict)
    C = defaultdict(nested_defaultdict)
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.995
    total_reward = []
    buffer = PrioritizedReplayBuffer(capacity=10000, alpha=alpha)

    for i_episode in range(1, num_episodes + 1):
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** i_episode))
        beta = beta_start + (beta_end - beta_start) * (i_episode / num_episodes)
        target_policy = create_epsilon_greedy_policy(Q, epsilon, env.action_space)
        
        print(f"\nEpisode {i_episode}/{num_episodes}")
        
        episode = []
        obs, info = env.reset()
        if obs is None:
            continue
        else:
            state = tuple(obs)
            episode_reward = 0
            print(f"Initial state: {state}")
        
        for t in range(max_steps_per_episode):
            action, _ = behavior_policy(state)
            node = np.random.choice(len(env.action_space.nvec))
            sub_action = np.zeros(len(env.action_space.nvec), dtype=int)
            sub_action[node] = np.random.choice(env.action_space.nvec[node])
            next_state, reward, done, _, _ = env.step(sub_action)
            episode_reward += reward
            episode.append((state, sub_action, reward))
            print(f"Step {t}, Node: {node}, Sub-action: {sub_action}, Reward: {reward}, Next state: {next_state}, Done: {done}")
            
            state = tuple(next_state)
            print(f"Step {t}, Reward: {reward}")

            if done:
                print(f"Episode ended after {t+1} steps")
                break

        total_reward.append(episode_reward)

        G = 0.0
        W = 1.0
        for t in range(len(episode))[::-1]:
            state, action, reward = episode[t]
            state_ = tuple(state)
            action_ = tuple(action)
            G = discount_factor * G + reward
            C[state_][action_] += W
            buffer.push(G, (state, action, reward))
            if len(buffer.buffer) > 32:
                mini_batch, weights, indices = buffer.sample(32, beta)
                errors = []
                for (s, a, r), w in zip(mini_batch, weights):
                    s_ = tuple(s)
                    a_ = tuple(a)
                    Q[s_][a_] += w * (G - Q[s_][a_])
                    errors.append(abs(G - Q[s_][a_]))
                buffer.update_priorities(indices, errors)
            if not np.array_equal(action_, target_policy(state)):
                break
            _, action_probs = behavior_policy(state)
            W = W * 1.0 / action_probs

        soft_update(target_Q, Q, tau)
        if i_episode % 100 == 0 or i_episode == num_episodes:
            print("\nCurrent best policy at episode {}:".format(i_episode))
            for state in Q.keys():
                best_action = max(Q[state].keys(), key=lambda x: Q[state][x])
                print("State: {}, Best action: {}, Q-value: {}".format(state, best_action, Q[state][best_action]))

    return Q, target_policy, total_reward
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
    plt.savefig('MC_1000.png')
    plt.show()

env = OutageEnv()
random_policy = create_random_policy(env)

obs, _ = env.reset()
print(f"Initial observation: {obs}")
print(f"Action space dimensions: {env.action_space.nvec}")
action, action_probs = random_policy(obs)
print("Random action:", action)
print("Action probabilities:", action_probs)
print_action_space(env)

Q, policy, total_reward = mc_control_importance_sampling_hierarchical(env, num_episodes=1000, behavior_policy=random_policy)

plot_reward(range(len(total_reward)), total_reward, "回合奖励变化")