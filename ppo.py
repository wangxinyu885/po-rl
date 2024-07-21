import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import gym
from Grid2OpOld.grid2op.Environment.outage_env import OutageEnv
import matplotlib.pyplot as plt
class PPOActor(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(PPOActor, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.logits = layers.Dense(action_size)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.logits(x)

class PPOCritic(tf.keras.Model):
    def __init__(self, state_size):
        super(PPOCritic, self).__init__()
        self.dense1 = layers.Dense(128, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.value = layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(inputs)
        return self.value(x)

class PPOAgent:
    def __init__(self, state_size, action_size, action_space_ranges, gamma=0.99, clip_ratio=0.2, lr=0.0003):
        self.state_size = state_size
        self.action_size = action_size
        self.action_space_ranges = action_space_ranges
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.actor = PPOActor(state_size, action_size)
        self.critic = PPOCritic(state_size)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    def get_action(self, state):
        logits = self.actor(state)
        action_probs = tf.nn.softmax(logits)
        action = tf.random.categorical(logits, 1)
        action = action.numpy().flatten()  # 生成多维动作并展平
        action = [np.clip(action[i], 0, self.action_space_ranges[i] - 1) for i in range(len(action))]
        return np.array(action, dtype=np.int32)

    def compute_advantages(self, rewards, values, next_values):
        deltas = rewards + self.gamma * next_values - values
        advantages = (deltas - np.mean(deltas)) / (np.std(deltas) + 1e-8)
        return advantages
    
    def train(self, states, actions, rewards, next_states):
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        rewards = np.array(rewards)
        
        values = self.critic(states)
        next_values = self.critic(next_states)
        
        advantages = self.compute_advantages(rewards, values, next_values)
        
        with tf.GradientTape() as tape:
            logits = self.actor(states)
            action_probs = tf.nn.softmax(logits)
            action_indices = tf.range(len(actions)) * self.action_size + actions
            selected_action_probs = tf.gather(tf.reshape(action_probs, [-1]), action_indices)
            old_action_probs = tf.stop_gradient(selected_action_probs)
            
            ratios = selected_action_probs / old_action_probs
            surr1 = ratios * advantages
            surr2 = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
            actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        with tf.GradientTape() as tape:
            values = self.critic(states)
            critic_loss = tf.reduce_mean(tf.square(rewards + self.gamma * next_values - values))
        
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

def preprocess_state(state):
    """
    预处理状态，将其展平为一维数组。
    """
    flat_state = []

    def flatten(data):
        if isinstance(data, dict):
            for key in sorted(data.keys()):
                flatten(data[key])
        elif isinstance(data, (list, np.ndarray, tuple)):
            for item in data:
                flatten(item)
        else:
            if data is not None:
                try:
                    flat_state.append(float(data))  # 确保将所有元素转换为浮点数
                except ValueError as e:
                    print(f"ValueError: {e} for data: {data}")
                except TypeError as e:
                    print(f"TypeError: {e} for data: {data}")
    
    flatten(state)
    return np.array(flat_state, dtype=np.float32)

def train_ppo(env, agent, num_episodes, max_steps, batch_size=64):
    all_rewards = []
    for episode in range(num_episodes):
        state = env.reset()
        state = preprocess_state(state)
        states, actions, rewards, next_states = [], [], [], []
        episode_reward = 0
        for step in range(max_steps):
            action = agent.get_action(np.atleast_2d(state).astype('float32'))
            step_result = env.step(action)
            next_state, reward, done, info = step_result
            next_state = preprocess_state(next_state)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            state = next_state
            episode_reward += reward
            if done or len(states) >= batch_size:
                agent.train(states, actions, rewards, next_states)
                states, actions, rewards, next_states = [], [], [], []
            if done:
                break
        all_rewards.append(episode_reward)
        if episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes}, Reward: {episode_reward}")
    return all_rewards

def plot_rewards(rewards, title="Training Rewards"):
    plt.plot(rewards)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.savefig('ppo_rewards.png')
    plt.show()

# 创建环境
env = OutageEnv()  
state_size = np.prod(env.observation_space.shape)
action_size = env.action_space.shape[0]  # 对于 MultiDiscrete 动作空间
action_space_ranges = env.action_space.nvec
# 初始化PPO代理
agent = PPOAgent(state_size, action_size, action_space_ranges)

# 训练PPO代理
num_episodes = 1000
max_steps = 100
rewards = train_ppo(env, agent, num_episodes, max_steps)

# 绘制奖励变化图
plot_rewards(rewards, "PPO Training Rewards")
