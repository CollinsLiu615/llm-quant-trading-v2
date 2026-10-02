import numpy as np

def strategy(prices, news, position, cash):
    n = prices.shape[1]
    last = prices[-1]
    valid = last > 0

    # --- Returns and basic statistics ---
    ret = np.diff(np.log(prices), axis=0)

    # --- Market regime (bull/bear) ---
    look_mkt = min(60, ret.shape[0])
    mkt_ret = ret[-look_mkt:].mean()
    mkt_vol = ret[-look_mkt:].std() + 1e-9
    regime = np.tanh(mkt_ret / mkt_vol)
    bear = (1 - regime) / 2

    # Trend: last price vs 60-day moving average (excluding today)
    if ret.shape[0] >= 60:
        prev_avg = np.mean(prices[-61:-1], axis=0)
        trend = (last > prev_avg).astype(float)
    else:
        trend = np.ones(n)

    # Combined market score
    market_score = regime * 0.7 + trend * 0.3

    # Position scale
    if market_score < 0.3:
        pos_scale = 0.1
    elif market_score < 0.5:
        pos_scale = 0.3
    else:
        pos_scale = 0.2 + 0.8 * market_score

    # --- Individual stock features ---
    if ret.shape[0] >= 20:
        mom = np.exp(ret[-20:].sum(axis=0)) - 1
        vol = ret[-20:].std(axis=0) + 1e-9
    else:
        mom = np.exp(ret.sum(axis=0)) - 1
        vol = ret.std(axis=0) + 1e-9

    risk_adj = mom / vol

    # --- News sentiment ---
    pw = ['涨','利好','突破','增长','收购','业绩','上涨','涨停','大涨','超预期','回购','增持','分红','盈利','走强']
    nw = ['跌','利空','下跌','亏损','暴雷','违规','退市','减持','风险','暴跌','跌停','预警','爆仓','问询','立案']
    txt = ''.join(news)
    sent = sum(w in txt for w in pw) - sum(w in txt for w in nw)
    sent_b = 1.0 + 0.2 * np.tanh(sent)

    # --- Score and selection ---
    score = risk_adj * sent_b * pos_scale
    score = np.where(valid, score, -np.inf)
    score = np.where(score > 0, score, -np.inf)

    # Top 10 picks
    idx = np.argsort(-score)[:10]
    raw = score[idx]
    s = raw.sum()
    w = np.zeros(10)
    if s > 0:
        wr = raw / s
        ex = np.maximum(wr - 0.15, 0)
        wr = np.minimum(wr, 0.15)
        if ex.sum() > 0:
            wr[ex > 0] += ex[ex > 0] / np.count_nonzero(ex)
        w = wr

    full = np.zeros(n)
    full[idx] = w

    # --- Stop-loss: individual stock drawdown > 8% from recent peak ---
    if ret.shape[0] >= 10:
        peak = np.max(prices[-10:-1], axis=0)
        drawdown = (peak - last) / peak
        sl_mask = valid & (drawdown > 0.08)
        full[sl_mask] = 0.0

    # --- Market-level stop-loss ---
    if ret.shape[0] >= 60:
        market_index = np.mean(prices, axis=1)
        high60 = np.max(market_index[-60:])
        mkt_dd = (high60 - market_index[-1]) / high60
        if mkt_dd > 0.15:
            full *= 0.1

    # --- Reduce turnover ---
    cur_val = (position * last).sum() + cash
    if cur_val > 0:
        cur_weights = (position * last) / cur_val
        new_full = np.where(full > cur_weights + 0.05, full,
                            np.where(full < cur_weights - 0.05, full,
                                     cur_weights))
    else:
        new_full = full

    # --- Budget and normalization ---
    need = np.maximum(new_full * cur_val - position * last, 0).sum()
    if need > cash:
        new_full *= cash / (need + 1e-9)
    t = new_full.sum()
    if t > 1:
        new_full /= t

    return new_full