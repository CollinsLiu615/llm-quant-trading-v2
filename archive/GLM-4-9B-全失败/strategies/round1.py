import numpy as np
import math

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    
    # 计算过去 5 天的回报率
    returns = prices[-5:] / prices[-6:] - 1
    
    # 计算波动率
    volatility = np.std(returns, axis=0)
    
    # 根据波动率排序，选择波动率较低的股票
    sorted_indices = np.argsort(volatility)
    
    # 选择前 20 只股票
    selected_stocks = sorted_indices[:20]
    
    # 计算总资金
    total_value = cash + np.sum(position * prices[-1])
    
    # 计算每只股票的权重
    for i in selected_stocks:
        w[i] = 1 / len(selected_stocks)
    
    return w