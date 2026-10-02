def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    p_last = prices[-1]
    p5 = prices[-6] if n_days >= 6 else p_last
    p20 = prices[-21] if n_days >= 21 else prices[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        ret5 = p_last / p5 - 1
        ret20 = p_last / p20 - 1
        rets = prices[1:] / prices[:-1] - 1
    ret5[~np.isfinite(ret5)] = np.nan
    ret20[~np.isfinite(ret20)] = np.nan
    rets[~np.isfinite(rets)] = np.nan
    rets[rets > 10] = np.nan
    rets[rets < -0.9] = np.nan
    if n_days >= 21:
        vol20 = np.nanstd(rets[-20:], axis=0)
        max_abs = np.nanmax(np.abs(rets[-20:]), axis=0)
    else:
        vol20 = np.nanstd(rets, axis=0)
        max_abs = np.nanmax(np.abs(rets), axis=0)
    valid = np.isfinite(p_last) & (p_last > 0) & np.isfinite(ret5) & np.isfinite(ret20) & np.isfinite(vol20)
    valid = valid & (vol20 > 0.001) & (vol20 < 0.08) & (p_last > 1.0)
    st_mask = np.isfinite(max_abs) & (max_abs < 0.057) & (vol20 < 0.015)
    valid[st_mask] = False
    valid[np.abs(ret5) > 0.6] = False
    if np.sum(valid) < 5:
        return w
    sentiment = 0
    try:
        if news is not None and len(news) > 0:
            text = " ".join([str(x) for x in news if isinstance(x, str)])
            pos = sum(text.count(x) for x in ["利好","上涨","增长","突破","新高","回暖","向好","牛市","增持"])
            neg = sum(text.count(x) for x in ["利空","下跌","暴跌","风险","监管","处罚","暴雷","退市","减持","熊市","大跌"])
            sentiment = pos - neg
    except:
        sentiment = 0
    m_ret = np.nanmean(ret20[valid])
    if not np.isfinite(m_ret):
        m_ret = 0
    raw = -ret5 / (vol20 + 0.02) + 0.12 * ret20 / (vol20 + 0.02)
    v = raw[valid]
    lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
    raw = np.clip(raw, lo, hi)
    score = np.full(n_stocks, -np.inf)
    score[valid] = raw[valid]
    try:
        held = position > 1e-6
        if np.any(held & valid):
            std_s = np.nanstd(score[valid])
            if np.isfinite(std_s) and std_s > 0:
                score[held & valid] += 0.35 * std_s
    except:
        held = np.zeros(n_stocks, dtype=bool)
    # 保留持仓降低换手：持仓且排名前40则继续持有
    rank = np.argsort(np.argsort(-score))
    selected = []
    for i in np.where(held & valid)[0]:
        if rank[i] < 40:
            selected.append(int(i))
    for idx in np.argsort(-score):
        if len(selected) >= 15:
            break
        if valid[idx] and idx not in selected:
            selected.append(int(idx))
    if len(selected) == 0:
        return w
    total = 0.95
    if m_ret < -0.06:
        total = 0.2
    elif m_ret < -0.03:
        total = 0.5
    elif m_ret < 0:
        total = 0.7
    if sentiment < -2:
        total *= 0.5
    elif sentiment < 0:
        total *= 0.7
    mean_vol = np.nanmean(vol20[valid])
    if np.isfinite(mean_vol) and mean_vol > 0.035:
        total *= 0.7
    total = float(np.clip(total, 0.0, 0.95))
    if total < 0.05:
        return w
    per = total / len(selected)
    per = min(per, 0.2)
    for idx in selected:
        w[idx] = per
    s = np.sum(w)
    if s > 1:
        w /= s
    return w