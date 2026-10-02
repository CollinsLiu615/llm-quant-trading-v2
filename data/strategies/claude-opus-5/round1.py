import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape  # 60, 5300
    
    # 最新价格
    latest_price = prices[-1]
    valid = (latest_price > 0) & np.isfinite(latest_price)
    
    # 收益率因子
    ret_5 = np.zeros(n_stocks)
    ret_20 = np.zeros(n_stocks)
    ret_60 = np.zeros(n_stocks)
    
    valid_5 = valid & (prices[-6] > 0)
    valid_20 = valid & (prices[-21] > 0)
    valid_60 = valid & (prices[0] > 0)
    
    ret_5[valid_5] = prices[-1][valid_5] / prices[-6][valid_5] - 1
    ret_20[valid_20] = prices[-1][valid_20] / prices[-21][valid_20] - 1
    ret_60[valid_60] = prices[-1][valid_60] / prices[0][valid_60] - 1
    
    # 波动率因子
    returns = np.diff(prices[-21:], axis=0) / (prices[-21:-1] + 1e-10)
    volatility = np.std(returns, axis=0)
    volatility[~np.isfinite(volatility)] = 999
    
    # 均线趋势
    ma_5 = np.mean(prices[-5:], axis=0)
    ma_20 = np.mean(prices[-20:], axis=0)
    trend = np.zeros(n_stocks)
    valid_ma = (ma_20 > 0) & np.isfinite(ma_5)
    trend[valid_ma] = ma_5[valid_ma] / ma_20[valid_ma] - 1
    
    # 因子标准化
    def norm(x):
        x = x.copy()
        x[~np.isfinite(x)] = 0
        mean = np.mean(x[valid])
        std = np.std(x[valid])
        if std > 1e-8:
            x = (x - mean) / std
        x[~np.isfinite(x)] = -999
        return x
    
    # 多因子合成
    score = (
        0.30 * norm(ret_20) +       # 中期动量
        0.15 * norm(-ret_5) +       # 短期反转
        0.25 * norm(-volatility) +  # 低波动
        0.20 * norm(trend) +        # 趋势
        0.10 * norm(ret_60)         # 长期动量
    )
    
    score[~valid] = -999
    
    # 持仓股票加分（降低换手）
    score[position > 0] += 0.15
    
    # 选取前20只
    top_indices = np.argsort(score)[-20:]
    top_scores = score[top_indices]
    
    # 权重分配
    w = np.zeros(n_stocks)
    top_scores = top_scores - top_scores.min() + 0.01
    weights = top_scores / top_scores.sum() * 0.95
    weights = np.minimum(weights, 0.20)
    weights = weights / weights.sum() * 0.95
    
    w[top_indices] = weights
    
    return w