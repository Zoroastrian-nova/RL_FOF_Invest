# RL_FOF_Invest
A Project for FOF Investment by RL

项目结构介绍：

|

| —— data：存放数据表与图片的文件夹，原始数据包括基金收盘价数据(来自Wind)以及FF3因子数据(来自CSMAR)

| —— main.py: 主函数代码，运行main.py即可完成数据处理以及模型的训练与评估

| —— _env_.py：此处定义了模拟FOF基金投资的强化学习环境

| —— _data_.py: 此处定义了数据的读取与预处理方式，并且将数据分解为训练集样本与验证集样本

| —— _model_.py: 此处定义了DQN模型以及用于模拟基金投资的智能体
