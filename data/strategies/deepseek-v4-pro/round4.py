import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    if n_days < 30 or n_stocks < 10:
        return np.zeros(n_stocks)

    position = np.asarray(position, dtype=float)
    news = np.atleast_1d(news) if news is not None else np.array([''])
    prices = np.where(np.isfinite(prices), prices, 0.0)

    last = prices[-1]
    prev = prices[-2]
    valid = (last > 0) & (prev > 0) & (last >= 2.0) & (last <= 1000.0)
    valid &= np.all(prices[-20:] > 0, axis=0)

    total_asset = cash + np.sum(position * last)
    if total_asset <= 0:
        total_asset = 1e6
    current_w = np.where(last > 0, position * last / total_asset, 0.0)
    current_w = np.nan_to_num(current_w, 0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        ret_matrix = prices[1:] / prices[:-1] - 1.0
    ret_matrix = np.where(np.isfinite(ret_matrix), ret_matrix, 0.0)
    vp = (prices[:-1] > 0) & (prices[1:] > 0)
    mkt = np.array([np.median(ret_matrix[t, vp[t]]) if np.any(vp[t]) else 0.0 for t in range(n_days - 1)])
    mkt_idx = np.cumprod(1.0 + np.clip(mkt, -0.2, 0.2))

    if len(mkt_idx) < 21:
        return np.zeros(n_stocks)

    mkt_ret5 = mkt_idx[-1] / mkt_idx[-6] - 1 if len(mkt_idx) >= 6 else 0.0
    mkt_ret20 = mkt_idx[-1] / mkt_idx[-21] - 1
    mkt_ret60 = mkt_idx[-1] / mkt_idx[-60] - 1 if len(mkt_idx) >= 60 else mkt_ret20
    mkt_ma20 = np.mean(mkt_idx[-20:])
    mkt_ma60 = np.mean(mkt_idx[-60:]) if len(mkt_idx) >= 60 else mkt_ma20
    mkt_dd60 = mkt_idx[-1] / np.max(mkt_idx[-60:] if len(mkt_idx) >= 60 else mkt_idx[-20:]) - 1
    mkt_dd20 = mkt_idx[-1] / np.max(mkt_idx[-20:]) - 1
    mkt_vol = np.std(mkt[-20:]) if len(mkt) >= 20 else 0.0

    pos_words = ['利好','增长','突破','盈利','回购','降准','降息','支持','签约','中标','上涨','创新高','改革','政策','放宽','复苏','超预期','扭亏','分红','增持']
    neg_words = ['利空','下滑','亏损','处罚','调查','退市','风险','减持','暴跌','违约','下调','警告','衰退','收紧','监管','跌停','警示','诉讼','债务','危机']
    news_text = ' '.join([str(x) for x in news]).lower()
    sentiment = sum(news_text.count(w) for w in pos_words) - sum(news_text.count(w) for w in neg_words)
    sentiment = max(-3, min(3, sentiment))

    bull = (mkt_idx[-1] > mkt_ma20) and (mkt_ma20 > mkt_ma60) and (mkt_ret20 > 0.02)
    bear = (mkt_idx[-1] < mkt_ma20) and (mkt_ma20 < mkt_ma60) and (mkt_ret20 < 0.0)

    exposure = 0.70 if bull else (0.0 if (bear or mkt_dd60 < -0.10 or mkt_ret20 < -0.10) else 0.40)
    exposure = float(np.clip(exposure + 0.02 * sentiment, 0.0, 0.90))

    if mkt_dd20 < -0.06:
        exposure = min(exposure, 0.15)
    if mkt_ret5 < -0.03:
        exposure = min(exposure, 0.15)
    if mkt_vol > 0.03:
        exposure = min(exposure, 0.20)
    if mkt_dd60 < -0.06:
        exposure = min(exposure, 0.10)
    if bear or mkt_dd60 < -0.10 or mkt_ret20 < -0.10 or mkt_ret5 < -0.05:
        exposure = 0.0
    exposure = float(np.clip(exposure, 0.0, 0.90))

    with np.errstate(divide='ignore', invalid='ignore'):
        ret1 = last / prev - 1.0
        vol20 = np.std(ret_matrix[-20:], axis=0)
        mom20 = last / prices[-21] - 1.0
        mom5 = last / prices[-6] - 1.0
        high20 = np.max(prices[-20:], axis=0)
        dd20 = last / high20 - 1.0

    valid &= (vol20 > 1e-8) & np.isfinite(vol20) & (ret1 > -0.095) & (ret1 < 0.095)
    stop_hit = (position > 0) & (dd20 < -0.15)
    valid &= ~stop_hit

    def zscore(arr, mask):
        vals = arr[mask]
        if len(vals) < 3:
            return np.zeros_like(arr)
        mu, sigma = np.mean(vals), np.std(vals)
        out = np.zeros_like(arr)
        if sigma > 1e-8:
            out[mask] = (arr[mask] - mu) / sigma
        return out

    held = (position > 0).astype(float)
    score = 0.60 * zscore(mom20, valid) + 0.15 * zscore(mom5, valid) - 0.25 * zscore(vol20, valid) + 0.10 * zscore(dd20, valid) + 0.25 * held

    idx = np.where(valid)[0]
    if len(idx) == 0:
        return np.zeros(n_stocks)

    K = min(10, len(idx))
    top = idx[np.argsort(-score[idx])[:K]]
    w_target = np.zeros(n_stocks)
    if exposure > 0 and K > 0:
        unit_w = min(0.10, exposure / K)
        w_target[top] = unit_w
        if w_target.sum() > exposure:
            w_target *= exposure / w_target.sum()

    current_exposure = float(np.sum(current_w))
    target_exposure = float(np.sum(w_target))
    risk_off = (exposure <= 0.05) or (mkt_dd60 < -0.08) or (mkt_ret20 < -0.08) or (mkt_dd20 < -0.08 and current_exposure > 0.20)
    urgent = bool(np.any(stop_hit)) or risk_off
    rebalance = (n_days % 5 == 0)

    if not rebalance and not urgent:
        w = current_w.copy()
    elif urgent:
        w = np.zeros(n_stocks) if risk_off else current_w.copy()
        w[stop_hit] = 0.0
        if not risk_off and exposure > 0:
            s = np.sum(w)
            if s > 1e-8 and target_exposure < s:
                w = w / s * target_exposure
    else:
        diff = float(np.sum(np.abs(w_target - current_w)))
        if diff < 0.12:
            w = current_w.copy()
        else:
            alpha = 0.40 if diff < 0.25 else 0.60
            w = (1.0 - alpha) * current_w + alpha * w_target

    w = np.nan_to_num(w, 0.0)
    w = np.maximum(w, 0.0)
    w[stop_hit] = 0.0
    s = np.sum(w)
    if s > 1.0:
        w = w / s
    return w