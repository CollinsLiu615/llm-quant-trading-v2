import numpy as np
def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    if n_days < 21:
        return w
    with np.errstate(divide='ignore', invalid='ignore'):
        daily_ret = np.diff(prices, axis=0) / prices[:-1, :]
    daily_ret = np.nan_to_num(daily_ret, nan=0.0, posinf=0.0, neginf=0.0)
    recent_ret = daily_ret[-20:, :]
    vol = np.std(recent_ret, axis=0)
    vol = np.where(vol < 1e-6, 1e-6, vol)
    with np.errstate(divide='ignore', invalid='ignore'):
        momentum = prices[-1, :] / prices[-21, :] - 1
    momentum = np.nan_to_num(momentum, nan=0.0, posinf=0.0, neginf=0.0)
    valid = (prices[-1, :] > 0) & (prices[-21, :] > 0) & (vol > 0)
    score = np.full(n_stocks, -np.inf)
    score[valid] = momentum[valid] / vol[valid]
    top_idx = np.argsort(score)[-20:]
    top_idx = [i for i in top_idx if score[i] > -np.inf]
    n_top = len(top_idx)
    if n_top == 0:
        return w
    weight = min(1.0 / n_top, 0.2)
    w[top_idx] = weight
    return w