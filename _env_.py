import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from gym import spaces
class MultiFundInvestmentEnv:
    """
    多基金投资环境类, 用于模拟基金投资过程。
    
    参数:
    - data: 基金数据, 包含基金代码、日期、收盘价等信息。
    - columns: 数据列名, 用于指定数据中的特定列。
    - initial_amount: 初始投资金额, 默认为10000。
    - transaction_cost: 交易成本, 默认为0.01。
    - window_size: 观测窗口大小, 默认为1。
    """
    def __init__(self, data,columns, initial_amount=10000, transaction_cost=0.01, window_size = 1):
        # 初始化环境参数
        self.data = data
        self.columns = columns
        self.dates = data.index.get_level_values('Date').unique()
        self.funds = data.index.get_level_values('Code').unique()
        self.initial_amount = initial_amount
        self.transaction_cost = transaction_cost
        self.current_date = 0
        self.current_fund = 0
        self.cash = initial_amount  # 初始现金
        self.portfolio_value = initial_amount  # 总资产
        self.holdings = {fund: 0 for fund in self.funds}
        self.previous_holdings_value = initial_amount  # 上一步持仓价值
        self.previous_close_prices = {}  # 上一步收盘价
        self.rewards = []
        self.returns = []
        self.values = []
        self.smoothed_reward = 1e-7
        self.window_size = window_size
        
        # 观测空间：每个基金的状态（如收益率、波动率等）
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.funds) * len(self.columns),), dtype=np.float32)
        
        # 动作空间：每个基金的投资比例（例如买入/卖出）
        self.action_space = spaces.MultiDiscrete([3] * len(self.funds))  # 假设每个基金的动作空间为买、卖、持有
        
    def reset(self):
        """
        重置环境到初始状态。
        
        返回:
        - observation: 初始观测值。
        """
        # 重置环境到初始状态
        self.current_date = 0
        self.current_fund = 0
        self.portfolio_value = self.initial_amount
        self.cash = self.initial_amount
        self.holdings = {fund: 0 for fund in self.funds}
        self.rewards = []
        self.returns = [1e-7]
        self.values = []
        self.smoothed_reward = 1e-7
        return self._get_observation(n= self.window_size)
    
    def step(self, action):
        """
        执行一步操作。
        
        参数:
        - action: 动作列表, 表示对每个基金的操作。
        
        返回:
        - observation: 新的观测值。
        - reward: 奖励。
        - done: 是否完成。
        - info: 信息字典, 包含每个基金的代码、持仓、收盘价和动作。
        """
        # 执行一步操作
        #if not isinstance(action, list) or len(action) != len(self.funds):
        #    raise ValueError("Action must be a list of the same length as the number of funds.")
        self._execute_action(action)
        self.current_date += 1
        self.current_fund = (self.current_fund + 1) % len(self.funds)
        
        
        # 计算新的观测值、奖励、是否完成以及信息字典
        observation = self._get_observation(n = self.window_size)
        

        reward = self._calculate_reward()
        # 更新 previous_holdings_value 为最新的 portfolio_value
        self.previous_holdings_value = self.portfolio_value
        # 记录奖励
        #log_return = np.log(self.portfolio_value + 1e-7) - np.log(self.previous_holdings_value+ 1e-7)
        #self.rewards.append(log_return)

        done = ((self.current_date >= (len(self.dates) - 1)) or (self.portfolio_value  <= 0))
        info = {(fund,self.holdings[fund],self.data.loc[(self.dates[self.current_date], fund), 'Close']) for fund in self.funds}
        info = pd.DataFrame(info,columns=["Code","Holdings","Close"])
        info["Action"] = action
         
        return observation, reward, done, info
    
    def _execute_action(self, action):
        """
        执行具体的投资动作。
        
        参数:
        - action: 动作列表, 表示对每个基金的操作。
        """
        date_time = self.dates[self.current_date]
        # 更新前一时刻的持仓价值和收盘价
        self.previous_holdings_value = self.portfolio_value
        self.close_prices = {fund: self.data.loc[(date_time, fund), 'Close'] for fund in self.funds}
        # 根据动作更新持仓
        for fund, act in enumerate(action):
            fund_name = self.funds[fund]
            last_close = self.data.loc[(date_time, fund_name), 'LastClose']
            if fund_name not in self.holdings:
                # 如果基金名称不在 holdings 中, 则添加一个键并初始化为 0
                self.holdings[fund_name] = 0

            try:
                close_price = self.data.loc[(date_time, fund_name), 'Close']
            except KeyError:
                close_price = 1e-7  # 或者设置其他默认值
            
            #if (close_price<= 1e-6):
            #    act = 2    
            
            if (act == 0) and (close_price> 1e-6):  # 买入
                self.buy(0.025,close_price,fund_name)
            
            
            elif (act == 1) and (close_price> 1e-6):  # 卖出
                self.sell(0.2,close_price=close_price,fund_name=fund_name)
            

            
            if (close_price<= 0.2) and (last_close <= 0.2):
                self.sell(1,close_price,fund_name) 
            
        # 更新持仓价值
        self.portfolio_value = self._calculate_value()
        self.values.append(self.portfolio_value)

    def buy(self,amount,close_price,fund_name):   
        """
        买入指定基金。
        
        参数:
        amount (float): 买入金额比例。
        close_price (float): 基金的收盘价。
        fund_name (str): 基金名称。
        """
        amount_to_invest  = min(self.cash * amount , self.cash)  # 确保不会超过现有现金
        amount_to_invest = max(amount_to_invest, 0)

        shares_to_buy = int(amount_to_invest / close_price)
        self.holdings[fund_name] += shares_to_buy
        self.cash -= amount_to_invest * (1 + self.transaction_cost)
    
    def sell(self,amount,close_price,fund_name):
        """
        卖出指定基金。
        
        参数:
        amount (float): 卖出份额比例。
        close_price (float): 基金的收盘价。
        fund_name (str): 基金名称。
        """
        amount_to_sell = np.floor(self.holdings[fund_name] * amount)  # 假设卖出份额为当前持有份额的25%
        shares_to_sell = min(amount_to_sell, self.holdings[fund_name])  # 确保不会卖出比持有的多 
        shares_to_sell = max(shares_to_sell, 0)
                
        self.holdings[fund_name] -= shares_to_sell
        self.cash += shares_to_sell * close_price * (1 - self.transaction_cost)
                
    def _get_observation(self, n=5):  # n 表示包含当前日期在内的前 n 天数据
        """
        获取当前及过去几天的基金状态。
        
        参数:
        n (int): 包含当前日期在内的天数。
        
        返回:
        np.array: 展平后的数据数组。
        """
        # 获取当前日期及之前 n 天内所有基金的状态
        start_date_index = max(0, self.current_date - n + 1)
        end_date_index = self.current_date
        
        # 获取指定范围内的所有日期
        dates_range = self.dates[start_date_index:end_date_index + 1]
        
        # 根据日期范围获取数据
        current_data = self.data.loc[pd.IndexSlice[dates_range, :], self.columns]

        # 计算需要填充的天数
        num_days_to_fill = n - len(dates_range)

        # 创建填充数据
        fill_data = np.full((num_days_to_fill*len(self.funds), len(self.columns)), 1e-7)
        if num_days_to_fill > 0:
            current_data = pd.concat([pd.DataFrame(fill_data, columns=self.columns), current_data], axis=0)

        # 确保 current_data 的形状为 (n, state_size)
        assert current_data.shape == (n*len(self.funds), len(self.columns)), f"Current data shape {current_data.shape} does not match expected {(n*len(self.funds), len(self.columns))}"
        # 将数据展平为一维数组
        flattened_data = current_data.values.flatten()

        return flattened_data
    
    def _calculate_value(self):
        """
        计算当前投资组合的价值。
        
        返回:
        float: 投资组合的价值。
        """
        holding_values = np.array(list(self.holdings.values()))
        close_prices_values = np.array(list(self.close_prices.values()))
        current_holdings_value = np.sum(np.array(holding_values) * np.array(close_prices_values))   
        return current_holdings_value   + self.cash  # 包括现金

    def _calculate_reward(self):

        """
        计算并返回奖励。
        
        返回:
        float: 计算得到的奖励。
        """
        log_return = np.log(self.portfolio_value + 1e-7) - np.log(self.previous_holdings_value+ 1e-7)

        self.returns.append(log_return)
        # 使用标准化处理
        if len(self.returns) <= 2:
            norm_reward = log_return
        else:
            mean_return = np.mean(self.returns)
            std_return = np.std(self.returns)
            norm_reward = (log_return - mean_return) / (std_return + 1e-7)  # 防止除零错误

        # 使用EMA平滑
        if len(self.returns) <= 2:
            smoothed_reward = norm_reward
        else:
            alpha = 0.25  # 平滑因子
            smoothed_reward = alpha * norm_reward + (1 - alpha) * self.smoothed_reward
        self.smoothed_reward = smoothed_reward
        return log_return
    
    def visualize_rewards(self,eps):
        """
        可视化奖励。
            
        参数:
        eps (int): 当前的回合数。
        """
        fig,(ax1,ax2) = plt.subplots(nrows=2,ncols=1,figsize=(16, 10))
        y_1 = np.array(self.returns)
        ax1.plot(self.dates,y_1)
        ax1.hlines(y = 0,xmin=self.dates.min(),xmax=self.dates.max(),colors=["red"])
        ax1.set_xlabel('Trading Dates')
        ax1.set_ylabel('Portfolio Returns')
        # 设置x轴的日期范围
        ax1.set_xlim([self.dates.min(), self.dates.max()])
        ax1.set_ylim([y_1.min(), y_1.max()])
        ax1.xaxis.set_major_locator(ticker.MaxNLocator(12))
        ax1.set_title('Portfolio Returns Over Time')

        y_2 = np.array(self.values)
        ax2.plot(self.dates[1:],y_2)
        ax2.hlines(y = self.initial_amount,xmin=self.dates[1:].min(),xmax=self.dates[1:].max(),colors=["red"])
        ax2.set_xlabel('Trading Dates')
        ax2.set_ylabel('Portfolio Value')
        ax2.set_xlim([self.dates[1:].min(), self.dates[1:].max()])
        ax2.set_ylim([y_2.min(), y_2.max()])
        ax2.xaxis.set_major_locator(ticker.MaxNLocator(12))
        ax2.set_title('Portfolio Value Over Time')

        plt.legend()
        plt.show()
        fig.savefig(f"./data/{eps}.png")


def train(env, agent, episodes=100,batch_size = 256,training_mode = True, target_update_frequency=10):
    """
    在给定的环境中训练智能体。
    
    参数:
    env: 智能体所处的环境。
    agent: 在环境中学习和行动的智能体。
    episodes: 训练的轮数, 默认为100轮。
    batch_size: 每次训练时从记忆中随机抽取的样本数量, 默认为256。
    training_mode: 是否进行训练, 默认为True。如果为False, 则不进行训练, 只进行测试。
    target_update_frequency: 更新目标网络的频率, 默认为每10个回合更新一次。
    
    返回:
    agent: 训练后的智能体。
    info: 训练过程中收集的信息。
    """
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        round = 0
        info = []
        while not done:
            # 代理基于当前状态选择动作
            if round%20 == 0:
                # 代理基于当前状态选择动作
                action = agent.act(state)
            else:
                action = np.full((agent.fund_class),fill_value=2)
                
            round += 1
            # 执行选定的动作, 并接收下一个状态、奖励、是否完成标志
            next_state, reward, done, info_data = env.step(action)
            info.append(info_data)
            # 存储记忆
            agent.remember(state, action, reward, next_state, done)
            
            # 更新当前状态
            state = next_state
            
            # 累加奖励
            total_reward += reward
            
            if training_mode  and len(agent.memory) > batch_size:
                # 进行训练
                agent.replay(batch_size)

            if (round) % target_update_frequency == 0:
                agent.update_target_network()
            
        total_reward = total_reward  #/round
        #if training_mode:
        #    scheduler.step()  # 更新学习率
        # 每个 episode 结束时打印进度
        print(f"Episode {e + 1} of {episodes}, Total Reward: {total_reward}")
        # 可视化奖励
        env.visualize_rewards(e)
    if training_mode:
        # 如果想要保存模型权重
        agent.save("./dqn_fund.pth")
    
    return agent,info