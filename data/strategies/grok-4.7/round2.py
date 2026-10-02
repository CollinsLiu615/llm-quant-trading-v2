def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    px = np.where(np.asarray(prices, dtype=float) > 0, np.asarray(prices, dtype=float), np.nan)
    last = px[-1]
    valid = np.isfinite(last)
    pos = np.asarray(position, dtype=float)

    rets = px[1:] / px[:-1] - 1.0
    rets = np.clip(np.where(np.isfinite(rets), rets, np.nan), -0.35, 0.35)

    mom = px[-1] / px[-21] - 1.0
    rev = -(px[-1] / px[-6] - 1.0)
    vol = np.nanstd(rets[-20:], axis=0)
    last_ret = rets[-1]
    r5 = px[-1] / px[-6] - 1.0
    peak = np.nanmax(px[-10:], axis=0)
    dd = last / peak - 1.0
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
    stop = (r5 < -0.09) | (dd < -0.12) | (last_ret < -0.08)
    bad = (
        (~valid) | (~np.isfinite(mom)) | (~np.isfinite(rev)) | (~np.isfinite(vol))
        | (vol <= 1e-8) | (finite_cnt < 18) | near_up | stop | (np.abs(mom) > 1.5)
    )
    score = np.where(bad, -np.inf, score)

    fin = np.isfinite(rets[-20:])
    cnt = np.sum(fin, axis=1)
    mean_r = np.sum(np.where(fin, rets[-20:], 0.0), axis=1) / np.maximum(cnt, 1)
    mkt20 = float(np.sum(np.where(cnt > 0, mean_r, 0.0)))
    mkt5 = float(np.sum(np.where(cnt[-5:] > 0, mean_r[-5:], 0.0)))

    text = " ".join(str(x) for x in news) if news else ""
    pw = ("上涨", "利好", "增长", "复苏", "降准", "降息", "新高", "超预期", "盈利", "突破", "回暖", "增持")
    nw = ("下跌", "利空", "暴跌", "亏损", "退市", "立案", "调查", "减持", "爆雷", "违约", "暴雷", "崩盘")
    sent = 0
    for w_ in pw:
        sent += text.count(w_)
    for w_ in nw:
        sent -= text.count(w_)

    exposure = 0.90
    if mkt20 < -0.10 or mkt5 < -0.05 or sent <= -5:
        exposure = 0.0
    elif mkt20 < -0.04 or mkt5 < -0.025 or sent <= -2:
        exposure = 0.35
    elif mkt20 < -0.015 or sent < 0:
        exposure = 0.60
    elif sent >= 3 and mkt5 > 0.0:
        exposure = 0.98
    if exposure <= 0.0:
        return np.zeros(n_stocks)

    if mkt20 < 0.0:
        score = np.where(mom > 0.0, score, -np.inf)

    ok = np.isfinite(score)
    if not np.any(ok):
        return np.zeros(n_stocks)

    held = (pos > 1e-8) & ok
    med = float(np.median(score[ok]))
    keep = held & (score >= med)

    k = 8
    chosen = keep.copy()
    n_keep = int(np.sum(chosen))
    if n_keep > k:
        held_idx = np.where(chosen)[0]
        worst = held_idx[np.argsort(score[held_idx])[: n_keep - k]]
        chosen[worst] = False
        n_keep = k
    if n_keep < k:
        need = k - n_keep
        cand = np.where(ok & (~chosen))[0]
        if cand.size > 0:
            top = cand[np.argsort(score[cand])[::-1][:need]]
            chosen[top] = True

    idx = np.where(chosen)[0]
    if idx.size == 0:
        return np.zeros(n_stocks)

    w = np.zeros(n_stocks)
    w[idx] = min(exposure / float(idx.size), 0.20)
    total = float(np.sum(w))
    if total > 1.0:
        w /= total
    return w