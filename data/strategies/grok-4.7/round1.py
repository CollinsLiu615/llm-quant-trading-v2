def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    px = np.where(np.asarray(prices, dtype=float) > 0, np.asarray(prices, dtype=float), np.nan)
    last = px[-1]
    valid = np.isfinite(last)

    prev, curr = px[:-1], px[1:]
    rets = curr / prev - 1.0
    rets = np.clip(np.where(np.isfinite(rets), rets, np.nan), -0.35, 0.35)

    mom = px[-6] / px[-21] - 1.0
    rev = -(px[-1] / px[-6] - 1.0)
    vol = np.nanstd(rets[-20:], axis=0)
    last_ret = rets[-1]
    finite_cnt = np.sum(np.isfinite(px[-21:]), axis=0)

    def zscore(x):
        m = np.nanmean(x)
        s = np.nanstd(x)
        if (not np.isfinite(s)) or s < 1e-12:
            return np.zeros(n_stocks)
        z = (x - m) / s
        return np.clip(np.where(np.isfinite(z), z, 0.0), -3.0, 3.0)

    score = 0.50 * zscore(mom) + 0.30 * zscore(rev) + 0.20 * zscore(-vol)

    near_up = (
        (np.abs(last_ret - 0.05) < 0.002)
        | (np.abs(last_ret - 0.10) < 0.003)
        | (np.abs(last_ret - 0.20) < 0.004)
        | (np.abs(last_ret - 0.30) < 0.005)
    )
    bad = (
        (~valid)
        | (~np.isfinite(mom))
        | (~np.isfinite(rev))
        | (~np.isfinite(vol))
        | (vol <= 1e-8)
        | (finite_cnt < 18)
        | near_up
        | (np.abs(mom) > 1.5)
    )
    score = np.where(bad, -np.inf, score)

    pos = np.asarray(position, dtype=float)
    held = (pos > 0) & valid
    score = np.where(np.isfinite(score), score + np.where(held, 0.40, 0.0), score)

    parts = []
    if news:
        for item in news:
            parts.append(str(item))
    text = " ".join(parts)
    pos_words = ["上涨", "利好", "增长", "复苏", "降准", "降息", "新高", "超预期", "盈利", "突破", "回暖", "增持"]
    neg_words = ["下跌", "利空", "暴跌", "亏损", "退市", "立案", "调查", "减持", "爆雷", "违约", "暴雷", "崩盘"]
    sent = 0
    for w in pos_words:
        sent += text.count(w)
    for w in neg_words:
        sent -= text.count(w)

    r = rets[-5:]
    finite = np.isfinite(r)
    cnt = np.sum(finite, axis=1)
    mean_r = np.sum(np.where(finite, r, 0.0), axis=1) / np.maximum(cnt, 1)
    mkt_ret = float(np.sum(np.where(cnt > 0, mean_r, 0.0)))

    exposure = 0.95
    if sent <= -3 or mkt_ret < -0.04:
        exposure = 0.50
    elif sent < 0 or mkt_ret < -0.02:
        exposure = 0.75
    elif sent >= 3 and mkt_ret > 0.0:
        exposure = 0.98

    n_ok = int(np.sum(np.isfinite(score)))
    if n_ok == 0:
        return np.zeros(n_stocks)
    k = 20 if n_ok >= 20 else n_ok
    idx = np.argpartition(score, n_stocks - k)[-k:]
    idx = idx[np.isfinite(score[idx])]
    if idx.size == 0:
        return np.zeros(n_stocks)
    idx = idx[np.argsort(score[idx])[::-1]]

    ranks = np.arange(idx.size, 0, -1, dtype=float)
    ww = ranks / np.sum(ranks) * exposure
    ww = np.minimum(ww, 0.20)
    if np.sum(ww) > 1.0:
        ww = ww / np.sum(ww)

    w = np.zeros(n_stocks)
    w[idx] = ww
    w = np.clip(np.where(np.isfinite(w), w, 0.0), 0.0, 0.20)
    total = np.sum(w)
    if total > 1.0:
        w = w / total
    return w