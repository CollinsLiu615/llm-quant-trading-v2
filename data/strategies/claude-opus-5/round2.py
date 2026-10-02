import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    
    latest_price = prices[-1]
    valid = (latest_price > 0) & np.isfinite(latest_price)
    
    # 市场状态识别
    market_returns = []
    for i in range(1, min(21, n_days)):
        valid_t = (prices[-i-1] > 0) & (prices[-i] > 0)
        if valid_t.sum() > 100:
            ret = np.nanmean(prices[-i][valid_t] / prices[-i-1][valid_t] - 1)
            market_returns.append(ret)
    
    market_trend = np.mean(market_returns) if market_returns else 0
    market_vol = np.std(market_returns) if len(market_returns) > 1 else 0.02
    
    # 根据市场状态调整目标仓位
    if market_trend < -0.01 or market_vol > 0.03:
        target_position = 0.50
    elif market_trend < 0:
        target_position = 0.70
    else:
        target_position = 0.90
    
    # 因子计算
    ret_5 = np.zeros(n_stocks)
    ret_20 = np.zeros(n_stocks)
    ret_60 = np.zeros(n_stocks)
    
    valid_5 = valid & (prices[-6] > 0) if n_days >= 6 else valid
    valid_20 = valid & (prices[-21] > 0) if n_days >= 21 else valid
    valid_60 = valid & (prices[0] > 0)
    
    if n_days >= 6:
        ret_5[valid_5] = prices[-1][valid_5] / prices[-6][valid_5] - 1
    if n_days >= 21:
        ret_20[valid_20] = prices[-1][valid_20] / prices[-21][valid_20] - 1
    ret_60[valid_60] = prices[-1][valid_60] / prices[0][valid_60] - 1
    
    # 波动率
    lookback = min(21, n_days)
    returns = np.diff(prices[-lookback:], axis=0) / (prices[-lookback:-1] + 1e-10)
    volatility = np.std(returns, axis=0)
    volatility[~np.isfinite(volatility)] = 999
    
    # 均线趋势
    ma_5 = np.mean(prices[-min(5, n_days):], axis=0)
    ma_20 = np.mean(prices[-min(20, n_days):], axis=0)
    trend = np.zeros(n_stocks)
    valid_ma = (ma_20 > 0) & np.isfinite(ma_5)
    trend[valid_ma] = ma_5[valid_ma] / ma_20[valid_ma] - 1
    
    def norm(x):
        x = x.copy()
        x[~np.isfinite(x)] = 0
        if valid.sum() > 0:
            mean = np.mean(x[valid])
            std = np.std(x[valid])
            if std > 1e-8:
                x = (x - mean) / std
        x[~np.isfinite(x)] = -999
        return x
    
    # 多因子评分
    score = (
        0.30 * norm(ret_20) +
        0.15 * norm(-ret_5) +
        0.25 * norm(-volatility) +
        0.20 * norm(trend) +
        0.10 * norm(ret_60)
    )
    
    score[~valid] = -999
    
    # 止损：持仓股票如果亏损超过15%，大幅降分
    current_holdings = position > 0
    if current_holdings.any() and n_days >= 21:
        for i in range(n_stocks):
            if position[i] > 0 and prices[-21][i] > 0:
                hold_return = latest_price[i] / prices[-21][i] - 1
                if hold_return < -0.15:
                    score[i] -= 3.0
                elif hold_return < -0.10:
                    score[i] -= 1.0
    
    # 持仓加分（大幅提高以降低换手）
    score[current_holdings] += 0.8
    
    # 选股
    n_select = 15
    top_indices = np.argsort(score)[-n_select:]
    top_scores = score[top_indices]
    
    # 权重分配
    w = np.zeros(n_stocks)
    if len(top_indices) > 0:
        top_scores = top_scores - top_scores.min() + 0.01
        weights = top_scores / top_scores.sum() * target_position
        weights = np.minimum(weights, 0.15)
        weights = weights / weights.sum() * target_position
        w[top_indices] = weights
    
    return w