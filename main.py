import logging
import math
import random
import gymnasium as gym
from collections import deque
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np


# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, device='cpu'):
        """
        初始化优先经验回放缓冲区。
        :param capacity: 缓冲区的最大容量
        :param alpha: 优先级指数，控制优先级的权重（0 <= alpha <= 1）
        :param beta: 重要性采样权重指数，初始值（0 <= beta <= 1）
        :param beta_increment_per_sampling: 每次采样后 beta 的增量
        :param device: 存储设备（'cpu' 或 'cuda'）
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)  # 存储经验的缓冲区
        self.priorities = deque(maxlen=capacity)  # 存储每个经验的优先级
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.index = 0  # 当前写入位置的索引
        self.device = device

    def add(self, experience, priority):
        """
        向缓冲区中添加经验。
        :param experience: 经验元组 (state, action, reward, next_state, done)
        :param priority: 该经验的初始优先级
        """
        # 将数据直接存储在设备上（CPU 或 GPU）
        experience = tuple(map(lambda x: x.to(self.device), experience))
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.index] = experience
            self.priorities[self.index] = priority
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        """
        从缓冲区中采样一批经验。
        :param batch_size: 采样批次大小
        :return: 采样的经验、对应的索引和重要性采样权重
        """
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha  # 计算每个经验的采样概率
        probs /= probs.sum()  # 归一化

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)  # 根据概率采样
        samples = [self.buffer[idx] for idx in indices]

        # 计算重要性采样权重
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # 更新 beta 值
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """
        更新采样经验的优先级。
        :param indices: 需要更新优先级的经验索引
        :param priorities: 新的优先级
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority


class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size):
        """
        初始化 Transformer 模型。
        :param input_size: 输入特征的维度
        :param d_model: Transformer 模型的隐藏层维度
        :param nhead: 多头注意力机制的头数
        :param num_layers: Transformer 编码器的层数
        :param output_size: 输出动作的维度
        """
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)  # 输入嵌入层
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, norm_first=True),  # 层归一化在前
            num_layers
        )
        self.fc = nn.Linear(d_model, output_size)  # 输出层

    def forward(self, x):
        """
        前向传播。
        :param x: 输入状态，形状为 [batch_size, sequence_length, d_model]
        :return: 输出动作值，形状为 [batch_size, output_size]
        """
        x = self.embedding(x)  # 嵌入输入
        x = x.permute(1, 0, 2)  # 调整维度以适应 Transformer 的输入形状 [sequence_length, batch_size, d_model]
        x = self.transformer_encoder(x)  # 通过 Transformer 编码器
        x = x[-1, :, :]  # 取最后一个时间步的输出 [batch_size, d_model]
        x = self.fc(x)  # 进行全连接层处理
        return x


class Network:
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, lr, tau, gamma, device='cpu'):
        """
        初始化神经网络模块。
        :param input_size: 输入状态的维度
        :param d_model: Transformer 模型的隐藏层维度
        :param nhead: 多头注意力机制的头数
        :param num_layers: Transformer 编码器的层数
        :param output_size: 输出动作的维度
        :param lr: 学习率
        :param tau: 目标网络更新系数
        :param gamma: 折扣因子
        :param device: 存储设备（'cpu' 或 'cuda'）
        """
        self.device = device
        self.gamma = gamma  # 新增 gamma 参数
        self.q_net = TransformerModel(input_size, d_model, nhead, num_layers, output_size).to(device)
        self.v_net = TransformerModel(input_size, d_model, nhead, num_layers, output_size).to(device)
        self.v_net.load_state_dict(self.q_net.state_dict())  # 同步初始权重
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        self.tau = tau

    def update_target_network(self):
        """
        更新目标网络。
        """
        q_net_state_dict = self.q_net.state_dict()
        v_net_state_dict = self.v_net.state_dict()
        for key in q_net_state_dict:
            v_net_state_dict[key] = self.tau * q_net_state_dict[key] + (1 - self.tau) * v_net_state_dict[key]
        self.v_net.load_state_dict(v_net_state_dict)

    def train(self, batch):
        """
        训练 Q 网络。
        :param batch: 采样的一批经验
        """
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        rewards = torch.cat(rewards, dim=0).unsqueeze(1)
        next_states = torch.cat(next_states, dim=0)
        dones = torch.cat(dones, dim=0).unsqueeze(1)

        q_values = self.q_net(states).gather(1, actions)  # 计算当前 Q 值
        with torch.no_grad():
            target_values = self.v_net(next_states).max(1).values  # 计算目标 V 值
            target_q_values = rewards + self.gamma * (1 - dones) * target_values.unsqueeze(1)  # 使用 self.gamma
        loss = nn.functional.mse_loss(q_values, target_q_values)  # 计算损失
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()  # 更新参数

        self.lr_scheduler.step()  # 更新学习率


class Exploration:
    def __init__(self, epsi_high, epsi_low, decay, temperature, device='cpu'):
        """
        初始化探索策略模块。
        :param epsi_high: epsilon 的初始值
        :param epsi_low: epsilon 的最小值
        :param decay: epsilon 的衰减率
        :param temperature: 玻尔兹曼探索的温度参数
        :param device: 存储设备（'cpu' 或 'cuda'）
        """
        self.epsi_high = epsi_high
        self.epsi_low = epsi_low
        self.decay = decay
        self.temperature = temperature
        self.steps = 0
        self._epsilon_decay_factor = math.exp(-1.0 / self.decay)  # 计算 epsilon 衰减因子
        self.device = device

    def get_epsilon(self):
        """
        计算当前 epsilon 值。
        :return: 当前 epsilon 值
        """
        return self.epsi_low + (self.epsi_high - self.epsi_low) * (self._epsilon_decay_factor ** self.steps)

    def act(self, q_values):
        """
        根据探索策略选择动作。
        :param q_values: Q 网络输出的动作值，形状为 [batch_size, action_dim]
        :return: 选择的动作，形状为 [1, 1]
        """
        self.steps += 1
        epsilon = self.get_epsilon()

        if random.random() > epsilon:
            probabilities = torch.softmax(q_values / self.temperature, dim=1)  # 玻尔兹曼探索
            action = torch.multinomial(probabilities, 1)  # 根据概率选择动作
        else:
            action = torch.tensor(
                [[random.randint(0, q_values.size(1) - 1)]],
                device=self.device,
                dtype=torch.long  # 指定 dtype
            )
        return action


class Agent:
    def __init__(self, env, replay_buffer, network, exploration, gamma, batch_size, n_steps=3, sequence_length=10):
        """
        初始化 Agent。
        :param env: 环境对象
        :param replay_buffer: 经验回放缓冲区
        :param network: 神经网络模块
        :param exploration: 探索策略模块
        :param gamma: 折扣因子
        :param batch_size: 训练批次大小
        :param n_steps: n-step 回报的步数
        :param sequence_length: 时序数据的长度
        """
        self.env = env
        self.replay_buffer = replay_buffer
        self.network = network
        self.exploration = exploration
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.sequence_length = sequence_length  # 时序数据的长度
        self.n_step_buffer = deque(maxlen=n_steps)  # 初始化 n-step 缓冲区
        self.state_buffer = deque(maxlen=sequence_length)  # 初始化状态缓冲区

    def _process_n_step(self):
        """
        处理 n-step 回报。
        """
        if len(self.n_step_buffer) >= self.n_steps:
            # 获取初始状态和动作
            state, action, _, _, _ = self.n_step_buffer[0]  # 解包初始经验
            # 计算 n-step 回报
            reward = sum([exp[2] * (self.gamma ** i) for i, exp in enumerate(self.n_step_buffer)])
            # 获取最终状态和终止标志
            next_state = self.n_step_buffer[-1][3]  # 下一个状态
            done = self.n_step_buffer[-1][4]  # 终止标志
            # 存储经验
            self.replay_buffer.add((state, action, reward, next_state, done), priority=1.0)

    def _get_sequence_state(self, state):
        """
        将单步状态扩展为时序数据。
        :param state: 当前状态，形状为 [state_dim]
        :return: 时序数据，形状为 [sequence_length, state_dim]
        """
        self.state_buffer.append(state)  # 将当前状态添加到状态缓冲区
        while len(self.state_buffer) < self.sequence_length:  # 如果缓冲区不足，用当前状态填充
            self.state_buffer.append(state)
        sequence_state = torch.stack(list(self.state_buffer), dim=0)  # 组合成时序数据，形状为 [sequence_length, state_dim]
        return sequence_state

    def train(self, num_episodes):
        """
        训练 Agent。
        :param num_episodes: 训练的总回合数
        """
        try:
            for episode in range(num_episodes):
                observation, info = self.env.reset()  # 正确解包返回值
                state = torch.tensor(observation, dtype=torch.float32, device=self.network.device)
                sequence_state = self._get_sequence_state(state).unsqueeze(0)   # 将单步状态扩展为时序数据，再扩展为批量状态
                total_reward = 0

                for step in range(1000):
                    q_values = self.network.q_net(sequence_state)  # 使用时序数据作为输入
                    action = self.exploration.act(q_values)
                    # 确保正确解包 env.step() 的返回值
                    next_observation, reward, terminated, truncated, info = self.env.step(action.item())
                    done = terminated or truncated

                    next_state = torch.tensor(next_observation, dtype=torch.float32, device=self.network.device)
                    sequence_next_state = self._get_sequence_state(next_state).unsqueeze(0)  # 将下一个状态扩展为时序数据，再扩展为批量状态
                    reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.network.device)
                    done_tensor = torch.tensor([done], dtype=torch.int8, device=self.network.device)

                    # 将经验添加到 n-step 缓冲区
                    self.n_step_buffer.append((sequence_state, action, reward_tensor, sequence_next_state, done_tensor))
                    self._process_n_step()  # 处理 n-step 回报

                    sequence_state = sequence_next_state  # 更新时序状态
                    total_reward += reward

                    if len(self.replay_buffer.buffer) >= self.batch_size:
                        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
                        self.network.train(batch)
                        self.network.update_target_network()  # 更新目标网络

                    if done:
                        # 清空 n-step 缓冲区
                        while len(self.n_step_buffer) > 0:
                            self._process_n_step()
                            self.n_step_buffer.popleft()
                        break

                logging.info(f"Episode {episode + 1}, Total Reward: {total_reward}")
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")


if __name__ == '__main__':
    # 初始化环境和超参数
    env = gym.make('CartPole-v1', render_mode="human")
    config = {
        'input_size': env.observation_space.shape[0],
        'd_model': 32,
        'nhead': 4,
        'num_layers': 2,
        'output_size': env.action_space.n,
        'lr': 1e-2,
        'tau': 0.005,
        'gamma': 0.99,
        'epsi_high': 0.9,
        'epsi_low': 0.05,
        'decay': int(1e3),
        'capacity': int(1e4),
        'batch_size': int(1e2),
        'temperature': 1.0,
        'n_steps': 3,  # 新增 n-step 参数
        'sequence_length': 10,  # 新增 sequence-length 参数
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 初始化模块
    replay_buffer = ReplayBuffer(capacity=config['capacity'], device=config['device'])
    network = Network(
        input_size=config['input_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        output_size=config['output_size'],
        lr=config['lr'],
        tau=config['tau'],
        gamma=config['gamma'],
        device=config['device']
    )
    exploration = Exploration(
        epsi_high=config['epsi_high'],
        epsi_low=config['epsi_low'],
        decay=config['decay'],
        temperature=config['temperature'],
        device=config['device']
    )

    # 初始化 Agent 并开始训练
    agent = Agent(
        env=env,
        replay_buffer=replay_buffer,
        network=network,
        exploration=exploration,
        gamma=config['gamma'],
        batch_size=config['batch_size'],
        n_steps=config['n_steps'],  # 传递 n-step 参数
        sequence_length=config['sequence_length'],  # 传递 sequence-length 参数
    )
    agent.train(num_episodes=1000)
