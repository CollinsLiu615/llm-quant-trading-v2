def strategy(prices, news, position, cash):
    import numpy as np
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    if n_days < 22:
        return position * 0.9 if position is not None else w
    p_last = prices[-1]
    p5 = prices[-6]
    p20 = prices[-21]
    with np.errstate(divide='ignore', invalid='ignore'):
        ret5 = p_last / p5 - 1
        ret20 = p_last / p20 - 1
        rets = prices[1:] / prices[:-1] - 1
    rets[~np.isfinite(rets)] = np.nan
    rets[(rets > 5) | (rets < -0.9)] = np.nan
    vol20 = np.nanstd(rets[-20:], axis=0)
    max_abs = np.nanmax(np.abs(rets[-20:]), axis=0)
    valid = np.isfinite(p_last) & (p_last > 0) & np.isfinite(ret5) & np.isfinite(ret20) & np.isfinite(vol20)
    valid &= (vol20 > 0.003) & (vol20 < 0.06) & (p_last > 2) & (np.abs(ret5) < 0.4)
    valid &= ~(np.isfinite(max_abs) & (max_abs < 0.05) & (vol20 < 0.012))
    if np.sum(valid) < 5:
        return position * 0.7
    m_ret = np.nanmean(ret20[valid])
    m_ret5 = np.nanmean(ret5[valid])
    mv = np.nanmean(vol20[valid])
    sentiment = 0
    try:
        if news is not None and len(news) > 0:
            txt = " ".join(str(x) for x in news if isinstance(x, str))
            pos = sum(txt.count(k) for k in ["利好","上涨","增长","突破","新高","回暖","向好","牛市","增持"])
            neg = sum(txt.count(k) for k in ["利空","下跌","暴跌","风险","监管","处罚","暴雷","退市","减持","熊市","大跌"])
            sentiment = pos - neg
    except:
        sentiment = 0
    if m_ret < -0.07 or m_ret5 < -0.05:
        total = 0.0
    elif m_ret < -0.04:
        total = 0.15
    elif m_ret < -0.02:
        total = 0.35
    elif m_ret < 0:
        total = 0.55
    else:
        total = 0.75
    if sentiment < -3:
        total *= 0.3
    elif sentiment < -1:
        total *= 0.6
    if np.isfinite(mv) and mv > 0.035:
        total *= 0.5
    total = float(np.clip(total, 0, 0.75))
    if total < 0.05:
        return position * 0.6
    raw = -ret5 / (vol20 + 0.02) + 0.05 * ret20 / (vol20 + 0.02)
    v = raw[valid]
    lo, hi = np.nanpercentile(v, 5), np.nanpercentile(v, 95)
    raw = np.clip(raw, lo, hi)
    score = np.full(n_stocks, -np.inf)
    score[valid] = raw[valid]
    held = position > 1e-6
    if np.any(held & valid):
        s = np.nanstd(score[valid])
        if np.isfinite(s) and s > 0:
            score[held & valid] += 1.0 * s
    rank = np.argsort(np.argsort(-score))
    selected = []
    for i in np.where(held & valid)[0]:
        if rank[i] < 40:
            selected.append(int(i))
    for idx in np.argsort(-score):
        if len(selected) >= 10:
            break
        if valid[idx] and idx not in selected:
            if total < 0.3 and not held[idx]:
                continue
            selected.append(int(idx))
    if not selected:
        return position * 0.8
    per = min(total / len(selected), 0.12)
    for i in selected:
        w[i] = per
    w = 0.6 * position + 0.4 * w
    s = np.sum(w)
    if s > total and s > 0:
        w *= total / s
    delta = np.abs(w - position)
    w[delta < 0.005] = position[delta < 0.005]
    s = np.sum(w)
    if s > 1:
        w /= s
    return w