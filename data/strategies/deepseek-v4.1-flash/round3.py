import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    
    # 需要足够的数据来计算指标
    min_days = 60
    if n_days < min_days:
        return w
    
    # 计算日收益率
    with np.errstate(divide='ignore', invalid='ignore'):
        daily_ret = np.diff(prices, axis=0) / prices[:-1, :]
    daily_ret = np.nan_to_num(daily_ret, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 市场收益和累计
    market_ret = np.nanmean(daily_ret, axis=1)
    market_cum = np.cumprod(1.0 + market_ret)
    
    # 市场风险系数计算
    recent_cum = market_cum[-min(60, len(market_cum)):]
    peak = np.max(recent_cum)
    dd = market_cum[-1] / peak - 1.0
    trend_short = market_cum[-1] / market_cum[-min(20, len(market_cum))] - 1.0
    trend_long = market_cum[-1] / market_cum[-min(60, len(market_cum))] - 1.0
    market_vol = np.std(market_ret[-min(20, len(market_ret)):]) + 1e-6
    
    # 风险系数：不再完全空仓，最低保留10%仓位
    if dd < -0.15:
        risk_factor = 0.1
    elif dd < -0.10:
        risk_factor = 0.3
    elif trend_long < -0.05:
        risk_factor = 0.4
    elif trend_long < -0.02:
        risk_factor = 0.6
    else:
        risk_factor = 1.0
    
    # 波动率目标调整（放宽目标波动率）
    target_vol = 0.02
    vol_scale = min(1.0, target_vol / market_vol)
    total_scale = risk_factor * vol_scale
    total_scale = float(np.clip(total_scale, 0.1, 1.0))  # 最低10%仓位
    
    # 选股：使用动态窗口
    lookback = min(60, n_days - 1)
    if lookback < 20:
        return w
    
    # 计算波动率
    vol = np.std(daily_ret[-lookback:, :], axis=0)
    vol = np.where(vol < 1e-6, 1e-6, vol)
    
    # 动量
    with np.errstate(divide='ignore', invalid='ignore'):
        momentum = prices[-1, :] / prices[-lookback, :] - 1.0
    momentum = np.nan_to_num(momentum, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 近期回撤
    recent_prices = prices[-min(20, n_days):, :]
    peak_20 = np.max(recent_prices, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        stock_dd = prices[-1, :] / peak_20 - 1.0
    stock_dd = np.nan_to_num(stock_dd, nan=-1.0, posinf=0.0, neginf=-1.0)
    
    # 过滤条件：放宽，不要求正动量，但排除极端下跌和异常
    valid = (
        (prices[-1, :] > 0)
        & (prices[-lookback, :] > 0)
        & (vol > 0)
        & (stock_dd > -0.25)  # 放宽回撤限制
        & (momentum > -0.15)  # 允许轻微负动量
    )
    
    if not np.any(valid):
        return w
    
    # 评分：动量 / 波动率，并加入近期回撤惩罚
    score = np.full(n_stocks, -np.inf)
    score[valid] = momentum[valid] / vol[valid]
    score[valid] *= (1 + stock_dd[valid])  # stock_dd为负，回撤大分数更低
    
    # 选择前10名
    top_n = min(10, np.sum(valid))
    top_idx = np.argsort(score)[-top_n:]
    top_idx = [i for i in top_idx if np.isfinite(score[i])]
    n_top = len(top_idx)
    if n_top == 0:
        return w
    
    # 分配权重：等权，但总仓位由total_scale控制
    weight_per_stock = total_scale / n_top
    max_weight = 0.18
    weight_per_stock = min(weight_per_stock, max_weight)
    
    w[top_idx] = weight_per_stock
    return w