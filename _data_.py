import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
import os

# 从环境变量中获取文件路径
data_dir = os.getenv("DATA_DIR", "./data")
def calculate_regression(sample, valid_indices):
    valid_X = sample[['CAPM', 'SMB', 'HML']].values[valid_indices]
    valid_y = sample['Return'].values[valid_indices]
    
    ols = LinearRegression()
    ols.fit(valid_X, valid_y)
    
    drawdown = np.min(sample['Return'].values[valid_indices])
    drawdown = min(drawdown, 1e-7)
    
    return [ols.coef_[0], ols.coef_[1], ols.coef_[2], ols.intercept_, drawdown]

def get_sample_data(fund_data, code):
    sample = fund_data[fund_data['Code'] == code]
    if sample.empty:
        raise ValueError(f"No data found for code: {code}")
    sample = sample.sort_values(by='Date')
    return sample
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
    # 读取股票和债券基金数据, 合并基金数据
    equity_data = pd.read_excel(os.path.join(data_dir, "fund_equity.xlsx"))
    bond_data = pd.read_excel(os.path.join(data_dir, "fund_bond.xlsx"))
    fund_data = pd.concat((equity_data, bond_data), axis=0)

    # 清洗数据
    fund_data.dropna(subset=["代码", "日期", "收盘价(元)"], inplace=True)
    fund_data.drop_duplicates(inplace=True)

    # 选择特定列并重命名列名
    selected_columns = fund_data[["代码", "日期", "收盘价(元)"]]
    selected_columns.columns = ["Code", "Date", "Close"]

    # 获取交易日期和基金代码
    trading_dates = np.sort(fund_data["Date"].unique())
    fund_codes = fund_data["Code"].unique()

    # 处理收盘价数据
    x = fund_data["Close"].values 
    x = np.where(x>10,x/100,x)
    fund_data["Close"]  = x
    long_table = pd.DataFrame([(date,fundname) for date in trading_dates for fundname in fund_codes],columns=("Date","Code"))
    fund_data = pd.merge(left=fund_data,right=long_table,how="right",on=("Date","Code"))

    # 检查收盘价是否全部大于等于0
    if (selected_columns["Close"] < 0).any():
        raise ValueError("收盘价数据中存在负值，数据可能有误")

    fund_data["NAN"] = fund_data["Close"]<1e-6
    Nan = fund_data.groupby("Code")["NAN"].mean()
    fund_codes = fund_codes[Nan<0.25]
    
    factors = pd.read_excel(os.path.join(data_dir, "STK_MKT_THRFACDAY.xlsx"))
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

        if len(trading_dates) == 0:
            return []
        
        try:
            sample = get_sample_data(fund_data, code)
        except ValueError as e:
            print(e)
            return []
      
        results = []
        
        # 对每个交易日滚动计算回归模型
        for date in trading_dates:
            valid_indices = np.where(sample['Date'].values <= date)[0]
            
            if valid_indices.size > 0:
                regression_results = calculate_regression(sample, valid_indices)
                results.append([code, date] + regression_results)
        
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
    result_df.to_csv(os.path.join(data_dir, "Fund_Data.csv"))
    print(f"Data Loaded:{result_df.describe()}")
    return result_df

def handle_missing_values(data):
    data.loc[data["Nan"], "Close"] = np.nan
    data = data.interpolate(method='linear')
    data.fillna(method="ffill", inplace=True)
    data.fillna(1e-7, inplace=True)
    return data
def add_code_and_date_columns(data, code):
    data["Code"] = code
    data["Date"] = data.index.get_level_values('Date')
    return data
def split_data(data):
    """
    分割数据集为训练集和验证集，并对数据进行预处理。
    
    1. 获取所有唯一的交易日期并按顺序排序。
    2. 获取所有唯一的基金代码。
    3. 将交易日期分为训练集日期和验证集日期, 验证集占20%, 不打乱顺序。
    4. 根据日期将数据分为训练集和验证集。
    5. 选择需要的列，并设置日期和基金代码为索引。
    6. 标记收盘价过低的数据，并根据条件过滤基金代码。
    7. 对于每个基金代码，提取对应的数据，处理缺失值，然后拼接到总的DataFrame中。
    8. 将收益率和最大回撤率乘以100进行缩放。
    9. 返回处理后的训练集、验证集和基金代码。
    """
    try:
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


        # 初始化空列表来存储过滤后的数据
        train_data_list = []
        valid_data_list = []
        # 对于每一个基金代码，使用xs方法提取对应的数据，然后拼接到总的DataFrame中
        for code in fund_codes:
            train_data_code = train_data.xs(code, level='Code', axis=0)
            valid_data_code = valid_data.xs(code, level='Code', axis=0)
            
            train_data_code = handle_missing_values(train_data_code)
            valid_data_code = handle_missing_values(valid_data_code)

            train_data_code = add_code_and_date_columns(train_data_code, code)
            valid_data_code = add_code_and_date_columns(valid_data_code, code)

            train_data_list.append(train_data_code)
            valid_data_list.append(valid_data_code)

        # 拼接数据
        train_data_filtered = pd.concat(train_data_list)
        valid_data_filtered = pd.concat(valid_data_list)

        # 将收益率和最大回撤率乘以100进行缩放
        train_data_filtered[["Return","MaxDrawdown"]] *= 100
        valid_data_filtered[["Return","MaxDrawdown"]] *= 100
        # 更新训练集和验证集
        train_data = train_data_filtered
        valid_data = valid_data_filtered
        # 设置索引
        train_data.set_index(["Date","Code"],inplace=True)
        valid_data.set_index(["Date","Code"],inplace=True)

        print(f"Training Dataset:{train_data.describe()}")
        print(f"Validation Dataset:{valid_data.describe()}")
        # 返回处理后的数据
        return train_data,valid_data,fund_codes
    
    except KeyError as e:
        print(f"KeyError: {e} - 请检查输入数据是否包含所有必要的列。")
        return None, None, None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None, None