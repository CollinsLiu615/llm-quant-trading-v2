import numpy as np
def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    min_days = 80
    if n_days < min_days:
        return w
    
    with np.errstate(divide='ignore', invalid='ignore'):
        daily_ret = np.diff(prices, axis=0) / prices[:-1, :]
    daily_ret = np.nan_to_num(daily_ret, nan=0.0, posinf=0.0, neginf=0.0)
    
    # === 1. Market regime detection ===
    market_ret = np.nanmean(daily_ret, axis=1)
    market_cum = np.cumprod(1.0 + market_ret)
    recent_cum = market_cum[-60:]
    peak = np.max(recent_cum)
    dd = market_cum[-1] / peak - 1.0
    trend_short = market_cum[-1] / market_cum[-20] - 1.0
    trend_long = market_cum[-1] / market_cum[-60] - 1.0
    market_vol = np.std(market_ret[-20:]) + 1e-6
    
    # === 2. Bear market defense: go to cash ===
    if dd < -0.10:
        return w
    if trend_long < -0.05 and trend_short < 0:
        return w
    if trend_long < -0.02 and market_vol > 0.025:
        return w
    
    # === 3. Regime scaling (bearish -> reduce exposure) ===
    if trend_long < -0.02:
        scale = 0.3
    elif trend_short < -0.01:
        scale = 0.5
    elif trend_short < 0:
        scale = 0.75
    else:
        scale = 1.0
    
    # Volatility targeting
    target_vol = 0.015
    vol_scale = min(1.0, target_vol / market_vol)
    scale *= vol_scale
    scale = float(np.clip(scale, 0.0, 1.0))
    
    # === 4. Stock selection with longer, more stable window ===
    lookback = 60
    vol = np.std(daily_ret[-lookback:, :], axis=0)
    vol = np.where(vol < 1e-6, 1e-6, vol)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        momentum = prices[-1, :] / prices[-lookback, :] - 1.0
    momentum = np.nan_to_num(momentum, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Individual stock stop-loss: recent drawdown filter
    recent = prices[-20:, :]
    peak_20 = np.max(recent, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        stock_dd = prices[-1, :] / peak_20 - 1.0
    stock_dd = np.nan_to_num(stock_dd, nan=-1.0, posinf=0.0, neginf=-1.0)
    
    # === 5. Filter and score ===
    valid = (
        (prices[-1, :] > 0)
        & (prices[-lookback, :] > 0)
        & (vol > 0)
        & (momentum > 0.02)
        & (stock_dd > -0.15)
    )
    score = np.full(n_stocks, -np.inf)
    score[valid] = momentum[valid] / vol[valid]
    
    top_idx = np.argsort(score)[-8:]
    top_idx = [i for i in top_idx if np.isfinite(score[i])]
    n_top = len(top_idx)
    if n_top == 0:
        return w
    
    weight = min(scale / n_top, 0.18)
    w[top_idx] = weight
    return w