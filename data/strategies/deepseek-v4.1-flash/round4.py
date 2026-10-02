import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    
    if n_days < 80:
        return w
    
    # ---------- 市场状态识别 ----------
    with np.errstate(divide='ignore', invalid='ignore'):
        daily_ret = np.diff(prices, axis=0) / prices[:-1, :]
    daily_ret = np.nan_to_num(daily_ret, nan=0.0, posinf=0.0, neginf=0.0)
    market_ret = np.nanmean(daily_ret, axis=1)
    market_cum = np.cumprod(1.0 + market_ret)
    
    ma20 = np.mean(market_cum[-20:])
    ma60 = np.mean(market_cum[-60:])
    trend_long = market_cum[-1] / market_cum[-60] - 1.0
    
    recent = market_cum[-60:]
    peak = np.max(recent)
    dd = market_cum[-1] / peak - 1.0
    
    # 熊市/风险状态：空仓
    bear = (dd < -0.12) or (trend_long < -0.08) or (ma20 < ma60 * 0.97)
    if bear:
        return w
    
    # 仓位系数
    if ma20 > ma60 and trend_long > 0.0:
        risk_factor = 1.0
    elif ma20 > ma60:
        risk_factor = 0.7
    else:
        risk_factor = 0.4
    
    # ---------- 个股选择 ----------
    ma20_stock = np.mean(prices[-20:, :], axis=0)
    ma60_stock = np.mean(prices[-60:, :], axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mom60 = prices[-1, :] / prices[-60, :] - 1.0
    mom60 = np.nan_to_num(mom60, nan=-1.0, posinf=0.0, neginf=-1.0)
    
    # 趋势过滤：多头排列 + 正动量
    valid = (
        (prices[-1, :] > 0)
        & (prices[-60, :] > 0)
        & (ma20_stock > ma60_stock)
        & (prices[-1, :] > ma20_stock)
        & (mom60 > -0.05)
    )
    
    if np.sum(valid) < 3:
        return w
    
    # 评分：动量强度
    score = np.full(n_stocks, -np.inf)
    score[valid] = mom60[valid]
    
    top_n = min(8, int(np.sum(valid)))
    top_idx = np.argsort(score)[-top_n:]
    
    target_w = np.zeros(n_stocks)
    target_w[top_idx] = 1.0 / top_n
    target_w *= risk_factor
    
    # ---------- 换手平滑 ----------
    current_w = np.zeros(n_stocks)
    if position is not None and len(position) == n_stocks:
        pos_val = np.asarray(position, dtype=float) * prices[-1, :]
        total_val = cash + np.sum(pos_val)
        if total_val > 0:
            current_w = pos_val / total_val
    
    alpha = 0.4
    w = alpha * target_w + (1 - alpha) * current_w
    
    s = np.sum(w)
    if s > 1.0:
        w = w / s
    
    return w