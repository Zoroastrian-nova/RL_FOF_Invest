

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
class DQN(nn.Module):
    """
    Deep Q-Network (DQN) 模型定义。
    
    参数:
    - state_size: 状态空间大小
    - action_size: 动作空间大小
    - asset_size: 资产类别数量
    - n: 输入状态的窗口大小, 默认为5
    """
    def __init__(self, state_size,action_size,asset_size, n=5):
        super(DQN, self).__init__()
        self.ln1 = nn.LayerNorm(n * state_size)
        self.fc1 = nn.Linear(n*state_size, 32)
        self.ln2 = nn.LayerNorm(32)
        self.fc2 = nn.Linear(32, 64)
        self.fc4 = nn.Linear(64, asset_size*action_size)
        self.dropout = nn.Dropout(p=0.33)  # 设置 dropout 概率

    def forward(self, x):
        """
        DQN模型的前向传播函数。
        
        参数:
        - x: 输入状态张量
        
        返回:
        - x: 经过网络处理后的张量
        """
        x = self.ln1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 应用 dropout
        x = self.ln2(x)
        assert not torch.isnan(x).any(), "FC1 output contains NaN values"
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)  # 应用 dropout
        #x = self.ln3(x)
        #assert not torch.isnan(x).any(), "FC2 output contains NaN values"
        #x = torch.relu(self.fc3(x))
        #assert not torch.isnan(x).any(), "FC3 output contains NaN values"
        x = self.fc4(x)
        return x

class DQNAgent:
    """
    DQN智能体定义。
    
    参数:
    - state_size: 状态空间大小
    - action_size: 动作空间大小
    - asset_size: 资产类别数量
    - n: 输入状态的窗口大小, 默认为5
    """
    def __init__(self, state_size, action_size,asset_size, n=5):
        self.state_size = state_size
        self.fund_class = asset_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.learning_rate = 0.01
        self.model = DQN(state_size,action_size,asset_size,n).to(self.device)
        weight_decay = 0.1 # L2 正则化系数
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        self.window_size = n

        self.target_model = DQN(state_size, action_size, asset_size,n).to(self.device)
        self.update_target_network()

    def update_target_network(self):
        """
        更新目标网络的权重, 使其等于当前模型的权重。
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        """
        存储经验到记忆池中。
        
        参数:
        - state: 当前状态
        - action: 执行的动作
        - reward: 获得的奖励
        - next_state: 下一个状态
        - done: 是否完成标志
        """
        # 确保 state 和 next_state 是一维数组
        state = state.flatten() if isinstance(state, np.ndarray) else state
        next_state = next_state.flatten() if isinstance(next_state, np.ndarray) else next_state

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        根据当前状态选择动作。
        
        参数:
        - state: 当前状态
        
        返回:
        - actions: 选择的动作数组
        """
        if np.random.rand() <= self.epsilon:
            random_actions = np.random.choice([0, 1, 2], size=self.fund_class, p=[0.2, 0.2, 0.6])
            return random_actions
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        if torch.isnan(state).any():
            raise ValueError("State contains NaN values")
        
        with torch.no_grad():
            q_values = self.model(state)
        q_values = q_values.view(1, self.fund_class, self.action_size)  # 转换为二维张量
        actions = q_values.argmax(dim=2).squeeze().cpu().numpy()

        return actions

    def replay(self, batch_size):
        """
        经验回放, 用于训练模型。
        
        参数:
        - batch_size: 小批量大小
        """
        if len(self.memory) < batch_size:
            return
        minibatch_index = np.random.randint(0,high=len(self.memory),size=batch_size)
        minibatch = [self.memory[i] for i in minibatch_index]
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # 检查 states 和 next_states 的形状
        #assert states.shape == (batch_size, self.window_size*self.state_size), f"States shape {states.shape} does not match expected {(batch_size, self.window_size*self.state_size)}"
        #assert next_states.shape == (batch_size, self.window_size*self.state_size), f"Next states shape {next_states.shape} does not match expected {(batch_size, self.window_size*self.state_size)}"

        # 确保 states 和 next_states 是二维数组
        states = np.array(states)
        next_states = np.array(next_states)
        
        states = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)  # 调整为与 states 相同的维度
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        # 确保 actions 是二维张量
        actions = actions.squeeze()  # 移除多余的维度
        
        # 创建 batch_indices 张量
        batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1)
        
        # 将 actions 转换为适合 gather 的形状
        actions = torch.cat([batch_indices, actions], dim=1)
        
        current_q_values = self.model(states)
        selected_q_values = current_q_values.gather(1, actions[:, 1].unsqueeze(1))
        
        max_next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        expected_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values
        
        loss = self.criterion(selected_q_values, expected_q_values)
        assert not torch.isnan(loss), "Loss contains NaN values"
        self.optimizer.zero_grad()
        loss.backward()
        #grads = [param.grad for param in self.model.parameters()]
        #for grad in grads:
        #    assert not torch.isnan(grad).any(), "Gradient contains NaN values"
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        """
        保存模型权重到文件。
        
        参数:
        - filename: 保存文件名
        """
        torch.save(self.model.state_dict(), filename)

    def load(self, filename):
        """
        从文件加载模型权重。
        
        参数:
        - filename: 加载文件名
        """
        self.model.load_state_dict(torch.load(filename, map_location=self.device))