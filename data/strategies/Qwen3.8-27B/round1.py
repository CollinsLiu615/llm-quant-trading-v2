import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    
    # 计算最近20天的动量（收益率）
    lookback = min(20, n_days - 1)
    if lookback <= 0:
        return w
    
    recent = prices[-(lookback + 1):]  # shape (lookback+1, n_stocks)
    prices_last = recent[-1, :]
    prices_start = recent[0, :]
    
    # 避免除零
    valid = prices_start > 0
    momentum = np.zeros(n_stocks)
    momentum[valid] = (prices_last[valid] - prices_start[valid]) / prices_start[valid]
    
    # 计算波动率（标准差）
    returns = np.diff(recent, axis=0)
    volatility = np.std(returns, axis=0)
    volatility = np.where(volatility < 1e-8, 1e-8, volatility)
    
    # 夏普比率代理：动量 / 波动率
    sharpe_like = momentum / volatility
    
    # 只考虑正动量的股票
    sharpe_like[momentum <= 0] = -1
    
    # 选出前20只
    top_indices = np.argsort(sharpe_like)[::-1][:20]
    
    # 分配权重：等权 5%
    for idx in top_indices:
        w[idx] = 0.05
    
    # 归一化确保 sum <= 1 且单只 <= 20%
    total = w.sum()
    if total > 1:
        w = w / total
    
    # 确保单只权重不超过 20%
    w = np.minimum(w, 0.20)
    
    # 再次归一化
    total = w.sum()
    if total > 1:
        w = w / total
    
    return w