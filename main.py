import os
import pandas as pd
from _data_ import load_data, split_data
from _env_ import MultiFundInvestmentEnv,train
from _model_ import DQNAgent
import matplotlib

def main():
    # 检查是否已经下载好了数据
    matplotlib.use('TkAgg')
    if os.path.exists('./data/Fund_Data.csv'):
        # 如果数据已下载, 则读取本地CSV文件
        data = pd.read_csv('./data/Fund_Data.csv')
    else:
        # 如果数据未下载, 则调用load_data函数加载数据
        data = load_data()
    train_data, valid_data, fund_codes = split_data(data)

    # 设置窗口大小, 用于确定环境中的观察窗口
    window_size = 10
    batch_size = 512
    col_list = ["CAPM_beta","SMB_beta","HML_beta","Alpha","Return",'MaxDrawdown','LastClose']
    # 初始化训练环境, 包含多个基金的投资环境, 用于训练模型
    env_train = MultiFundInvestmentEnv(data=train_data,columns=col_list, 
                                    initial_amount=10000, transaction_cost=0.001, window_size = window_size)

    # 初始化验证环境, 与训练环境相似, 用于验证模型的性能
    env_valid = MultiFundInvestmentEnv(data=valid_data,columns=col_list,
                                        initial_amount=10000, transaction_cost=0.001, window_size = window_size)

    # 初始化 DQN 代理
    state_size = env_train.observation_space.shape[0]
    action_size = env_train.action_space.nvec[0]  # 假设所有基金的动作空间相同
    agent = DQNAgent(state_size, action_size,asset_size = len(fund_codes),n = window_size)

    # 使用训练环境对代理进行训练
    agent,info_train = train(env_train, agent, episodes = 4,batch_size = batch_size)

    # 使用验证环境对代理进行测试, 不进行训练
    valid,info_valid = train(env_valid, agent, episodes = 1,batch_size = batch_size,training_mode=False)

    # 将训练信息保存到数据框中, 以便后续分析
    df_train_info = pd.concat(info_train,ignore_index=True)

    # 将验证信息保存到数据框中, 以便后续分析
    df_valid_info = pd.concat(info_valid,ignore_index=True)

    # 将训练信息导出到CSV文件
    df_train_info.to_csv("./data/df_train_info.csv",index=False)

    # 将验证信息导出到CSV文件
    df_valid_info.to_csv("./data/df_valid_info.csv",index=False)

if __name__ == '__main__':
    main()
