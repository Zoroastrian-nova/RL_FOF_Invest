import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
def load_data():
    """
    加载并处理基金数据和因子数据。
    
    该函数首先从Excel文件中读取股票和债券基金的数据, 然后合并这些数据并进行清洗。
    接着, 它读取并合并市场因子数据, 计算基金的收益率, 并对数据进行进一步的处理。
    最后, 它使用并行处理方法对每个基金进行回归分析, 并将结果保存到CSV文件中。
    
    Returns:
        pd.DataFrame: 包含所有处理后的基金数据和回归分析结果的DataFrame。
    """
    # 读取股票和债券基金数据, 合并基金数据
    data_1 = pd.read_excel("./data/fund_equity.xlsx")
    data_2 = pd.read_excel("./data/fund_bond.xlsx")
    fund_data = pd.concat((data_1, data_2),axis=0)

    # 选择特定列并重命名列名
    fund_data = fund_data[["代码","日期","收盘价(元)"]]
    fund_data.columns = ["Code","Date","Close"]

    # 获取交易日期和基金代码
    trading_dates = np.sort(fund_data["Date"].unique())
    fund_codes = fund_data["Code"].unique()

    # 处理收盘价数据
    x = fund_data["Close"].values 
    x = np.where(x>10,x/100,x)
    fund_data["Close"]  = x
    long_table = pd.DataFrame([(date,fundname) for date in trading_dates for fundname in fund_codes],columns=("Date","Code"))
    fund_data = pd.merge(left=fund_data,right=long_table,how="right",on=("Date","Code"))

    fund_data["NAN"] = fund_data["Close"]<1e-6
    Nan = fund_data.groupby("Code")["NAN"].mean()
    fund_codes = fund_codes[Nan<0.25]
    
    factors = pd.read_excel("./data/STK_MKT_THRFACDAY.xlsx")
    factors["Date"] = pd.to_datetime(factors["Date"])
    fund_data = pd.merge(fund_data, factors, on="Date",how="left")
    fund_data["Return"] = fund_data["Close"].transform(lambda x: np.log(x) - np.log(x.shift(1)))

    # 为缺失的收益率填充一个极小值
    fund_data["Return"] = fund_data["Return"].fillna(1e-7)

    # 定义一个函数来处理单个基金的数据
    def process_fund(code, trading_dates):
        """
        对单个基金进行数据处理和回归分析。
        
        Args:
            code (str): 基金代码。
            trading_dates (numpy.ndarray): 交易日期数组。
        
        Returns:
            list: 包含回归分析结果的列表。
        """
        sample = fund_data[fund_data['Code'] == code]
        sample = sample.sort_values(by='Date')
        X = sample[['CAPM', 'SMB', 'HML']].values
        y = sample['Return'].values
        dates = sample['Date'].values
        returns = sample['Return'].values
        
        results = []
        
        # 对每个交易日滚动计算回归模型
        for date_index, date in enumerate(trading_dates):
            # 截取到当前日期为止的数据
            valid_indices = np.where(dates <= date)[0]
            
            if valid_indices.size > 0:
                valid_X = X[valid_indices]
                valid_y = y[valid_indices]
                
                
                # 拟合模型
                ols = LinearRegression()
                ols.fit(valid_X, valid_y)

                
                drawdown = np.min(returns[valid_indices])
                drawdown = min(drawdown, 1e-7)
                # 保存结果
                list_data = [code, date, ols.coef_[0], ols.coef_[1], ols.coef_[2], ols.intercept_,drawdown]
                results.append(list_data)
        
        return results

    # 使用 joblib 并行处理
    num_jobs = -1  # 使用所有可用的核心
    results = Parallel(n_jobs=num_jobs)(delayed(process_fund)(code, trading_dates) for code in fund_codes)

    # 收集结果
    flat_results = [item for sublist in results for item in sublist]

    # 转换为 DataFrame
    result_df = pd.DataFrame(flat_results, columns=['Code', 'Date', 'CAPM_beta', 'SMB_beta', 'HML_beta', 'Alpha','MaxDrawdown'])
    result_df["Return"] = fund_data["Return"].shift(-1)
    result_df["Close"] = fund_data["Close"]
    result_df["LastClose"] = fund_data["Close"].shift(-1)
    result_df[["CAPM_beta","SMB_beta","HML_beta"]] = result_df[["CAPM_beta","SMB_beta","HML_beta"]] + 1e-7
    result_df = result_df.fillna(1e-7)
    result_df.to_csv("./data/Fund_Data.csv")
    return result_df


def split_data(data):
    # 获取所有唯一的交易日期并按顺序排序
    trading_dates = np.sort(data["Date"].unique())
    # 获取所有唯一的基金代码
    fund_codes = data["Code"].unique()

    # 将交易日期分为训练集日期和验证集日期, 验证集占20%, 不打乱顺序
    train_dates, valid_dates = train_test_split(trading_dates, test_size=0.2, shuffle=False)
    # 根据日期将数据分为训练集和验证集
    train_data,valid_data = data[data['Date']<=train_dates[-1]],data[data["Date"]>=valid_dates[0]]

    # 选择训练集中需要的列, 并设置日期和基金代码为索引
    train_data = train_data[["Date","Code","CAPM_beta","SMB_beta","HML_beta","Alpha","Return","Close",'MaxDrawdown','LastClose']]
    train_data = train_data.set_index(["Date","Code"])
    # 选择验证集中需要的列, 并设置日期和基金代码为索引
    valid_data = valid_data[["Date","Code","CAPM_beta","SMB_beta","HML_beta","Alpha","Return","Close",'MaxDrawdown','LastClose']]
    valid_data = valid_data.set_index(["Date","Code"])

    valid_data["Nan"] = valid_data["Close"]<1e-6
    train_data["Nan"] = train_data["Close"]<1e-6
    fund_codes = fund_codes[(train_data.groupby("Code")["Nan"].mean()<0.25) & (valid_data.groupby("Code")["Nan"].mean()<0.25)]


    # 初始化空DataFrame来存储过滤后的数据
    train_data_filtered = pd.DataFrame()
    valid_data_filtered = pd.DataFrame()
    # 对于每一个基金代码，使用xs方法提取对应的数据，然后拼接到总的DataFrame中
    for code in fund_codes:
        train_data_code = train_data.xs(code, level='Code', axis=0)
        valid_data_code = valid_data.xs(code, level='Code', axis=0)
        train_data_code[train_data_code["Nan"]] = np.nan
        valid_data_code[valid_data_code["Nan"]] = np.nan
        train_data_code = train_data_code.interpolate(method='linear')
        valid_data_code = valid_data_code.interpolate(method='linear')
        train_data_code.fillna(method="ffill",inplace=True)
        valid_data_code.fillna(method="ffill",inplace=True)
        train_data_code.fillna(1e-7,inplace=True)
        valid_data_code.fillna(1e-7,inplace=True)
        train_data_code["Code"],valid_data_code["Code"] = code,code
        train_data_code["Date"] = train_data_code.index.copy()
        valid_data_code["Date"] = valid_data_code.index.copy()
        
        train_data_filtered = pd.concat([train_data_filtered, train_data_code])
        valid_data_filtered = pd.concat([valid_data_filtered, valid_data_code])

    # 将收益率和最大回撤率乘以100进行缩放
    train_data_filtered[["Return","MaxDrawdown"]] *= 100
    valid_data_filtered[["Return","MaxDrawdown"]] *= 100

    train_data = train_data_filtered
    valid_data = valid_data_filtered
    train_data.set_index(["Date","Code"],inplace=True)
    valid_data.set_index(["Date","Code"],inplace=True)



    return train_data,valid_data,fund_codes