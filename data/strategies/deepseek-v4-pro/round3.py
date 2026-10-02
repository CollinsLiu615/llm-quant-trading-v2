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
    ref20 = prices[-21]
    ref5 = prices[-6]

    valid = (last > 0) & (prev > 0) & (ref20 > 0) & (ref5 > 0) & (last >= 2.0) & (last <= 1000.0)
    valid &= np.all(prices[-20:] > 0, axis=0)

    total_asset = cash + np.sum(position * last)
    if total_asset <= 0:
        total_asset = 1e6
    current_w = np.where(last > 0, position * last / total_asset, 0.0)
    current_w = np.where(np.isfinite(current_w), current_w, 0.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        ret_matrix = prices[1:] / prices[:-1] - 1.0
    ret_matrix = np.where(np.isfinite(ret_matrix), ret_matrix, 0.0)
    vp = (prices[:-1] > 0) & (prices[1:] > 0)
    mret = np.sum(ret_matrix * vp, axis=1) / np.maximum(np.sum(vp, axis=1), 1)
    mkt = np.cumprod(1.0 + mret)
    if len(mkt) < 21:
        return np.zeros(n_stocks)

    mkt_ret5 = mkt[-1] / mkt[-6] - 1.0 if len(mkt) >= 6 else 0.0
    mkt_ret20 = mkt[-1] / mkt[-21] - 1.0
    mkt_ma20 = np.mean(mkt[-20:])
    mkt_ma60 = np.mean(mkt[-60:]) if len(mkt) >= 60 else mkt_ma20
    mkt_dd = mkt[-1] / np.max(mkt[-60:] if len(mkt) >= 60 else mkt[-20:]) - 1.0

    pos_words = ['利好','增长','突破','盈利','回购','降准','降息','支持','签约','中标','上涨','创新高','改革','政策','放宽','复苏','超预期','扭亏','分红','增持']
    neg_words = ['利空','下滑','亏损','处罚','调查','退市','风险','减持','暴跌','违约','下调','警告','衰退','收紧','监管','跌停','警示','诉讼','债务','危机']
    news_text = ' '.join([str(x) for x in news]).lower()
    sentiment = sum(news_text.count(w) for w in pos_words) - sum(news_text.count(w) for w in neg_words)
    sentiment = max(-3, min(3, sentiment))

    exposure = 0.60
    if mkt_ret20 < 0:
        exposure -= 0.15
    if mkt_ret5 < -0.02:
        exposure -= 0.10
    if mkt[-1] < mkt_ma20:
        exposure -= 0.20
    if mkt_ma20 < mkt_ma60:
        exposure -= 0.15
    if mkt_dd < -0.05:
        exposure -= 0.15
    exposure += 0.02 * sentiment
    exposure = float(np.clip(exposure, 0.0, 1.0))
    if mkt_dd < -0.10 or mkt_ret20 < -0.10:
        exposure = min(exposure, 0.15)

    with np.errstate(divide='ignore', invalid='ignore'):
        ret1 = last / prev - 1.0
        vol = np.std(ret_matrix[-20:], axis=0)
        mom20 = last / ref20 - 1.0
        mom5 = last / ref5 - 1.0
        high20 = np.max(prices[-20:], axis=0)
        dd20 = last / high20 - 1.0

    valid &= (vol > 1e-8) & np.isfinite(vol) & (ret1 > -0.095) & (ret1 < 0.095)
    stop_hit = (position > 0) & (dd20 < -0.15)
    valid &= ~stop_hit

    def zscore(arr, mask):
        vals = arr[mask]
        mu = np.mean(vals)
        sigma = np.std(vals)
        out = np.zeros_like(arr)
        if sigma > 1e-8:
            out[mask] = (arr[mask] - mu) / sigma
        return out

    z20 = zscore(mom20, valid)
    z5 = zscore(mom5, valid)
    zv = zscore(vol, valid)
    held = (position > 0).astype(float)
    score = 0.55 * z20 + 0.15 * z5 - 0.25 * zv + 0.35 * held

    idx = np.where(valid)[0]
    if len(idx) == 0:
        if np.any(stop_hit) or exposure < 0.05:
            return np.zeros(n_stocks)
        return current_w

    K = min(10, len(idx))
    top = idx[np.argsort(-score[idx])[:K]]
    w_target = np.zeros(n_stocks)
    if exposure > 0 and K > 0:
        weight = min(0.10, exposure / K)
        w_target[top] = weight
        if w_target.sum() > exposure:
            w_target *= exposure / np.sum(w_target)

    current_exposure = np.sum(current_w)
    target_exposure = np.sum(w_target)
    rebalance = (n_days % 5 == 0)
    risk_off = (mkt_dd < -0.08) or (mkt_ret20 < -0.08) or (target_exposure < current_exposure - 0.20)
    urgent = np.any(stop_hit) or risk_off

    if not rebalance and not urgent:
        return current_w

    if urgent:
        w = np.zeros(n_stocks)
        if exposure >= 0.10 and mkt_dd >= -0.12:
            w = current_w.copy()
            w[stop_hit] = 0.0
            cur_sum = np.sum(w)
            if cur_sum > 1e-8 and target_exposure < cur_sum:
                w = w / cur_sum * target_exposure
    else:
        diff = np.sum(np.abs(w_target - current_w))
        if diff < 0.08:
            w = current_w
        else:
            alpha = 0.45 if diff < 0.15 else 0.70
            w = (1.0 - alpha) * current_w + alpha * w_target

    w = np.nan_to_num(w, 0.0)
    w = np.maximum(w, 0.0)
    w[stop_hit] = 0.0
    if np.sum(w) > 1.0:
        w = w / np.sum(w)
    return w