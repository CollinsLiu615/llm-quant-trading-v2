import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    
    # 市场状态识别
    lookback_market = min(60, n_days - 1)
    if lookback_market <= 1:
        return w
        
    recent_market = prices[-(lookback_market + 1):]
    market_return = (recent_market[-1, :] - recent_market[0, :]) / (recent_market[0, :] + 1e-8)
    avg_market_return = np.mean(market_return)
    
    # 熊市降仓系数
    if avg_market_return < -0.05:
        exposure = 0.3
    elif avg_market_return < 0:
        exposure = 0.7
    else:
        exposure = 1.0
        
    # 计算最近20天的动量
    lookback = min(20, n_days - 1)
    if lookback <= 0:
        return w
        
    recent = prices[-(lookback + 1):]
    prices_last = recent[-1, :]
    prices_start = recent[0, :]
    
    valid = prices_start > 0
    momentum = np.zeros(n_stocks)
    momentum[valid] = (prices_last[valid] - prices_start[valid]) / prices_start[valid]
    
    # 计算波动率
    returns = np.diff(recent, axis=0)
    volatility = np.std(returns, axis=0)
    volatility = np.where(volatility < 1e-8, 1e-8, volatility)
    
    # 夏普比率代理
    sharpe_like = momentum / volatility
    sharpe_like[momentum <= 0] = -1
    
    # 选出前20只
    top_indices = np.argsort(sharpe_like)[::-1][:20]
    
    # 分配权重
    for idx in top_indices:
        w[idx] = 0.05
    
    # 归一化
    total = w.sum()
    if total > 0:
        w = w / total
    
    # 应用市场暴露
    w = w * exposure
    
    # 限制单只权重不超过 20%
    w = np.minimum(w, 0.20)
    
    # 二次归一化确保 sum <= 1
    total = w.sum()
    if total > 1:
        w = w / total
        
    # 简单的持仓稳定性：如果当前持仓权重高于阈值且排名仍在前列，保持权重以减少换手
    # 这里简化处理，仅返回计算出的目标权重，换手率由回测引擎根据权重变化计算
    # 为了减少换手，我们可以引入一定的滞后或缓冲，但鉴于代码行数限制，
    # 我们依靠市场状态降低在熊市中的仓位波动，以及固定的20只选股池来间接控制。
    
    return w