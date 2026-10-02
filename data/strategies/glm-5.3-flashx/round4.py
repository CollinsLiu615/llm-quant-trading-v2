import numpy as np


def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    if n_stocks == 0 or n_days < 30:
        return w

    # ---------- 0. 当前持仓权重（非调仓日原样输出 => 零换手） ----------
    p = np.array(prices, dtype=np.float64)
    p[~np.isfinite(p)] = 0.0
    pos = np.asarray(position, dtype=np.float64)
    mv = np.where(p[-1] > 0.0, pos * p[-1], 0.0)
    equity = float(mv.sum() + max(float(cash), 0.0))
    cur_w = mv / equity if equity > 1e-9 else np.zeros(n_stocks)

    # ---------- 1. 数据清洗 ----------
    tradable = p[-1] > 0.0
    if n_days >= 3:
        tradable &= ~((p[-1] == p[-2]) & (p[-2] == p[-3]))   # 疑似停牌

    prev = p[:-1]
    ok = prev > 0.0
    r = np.where(ok, p[1:] / np.where(ok, prev, 1.0) - 1.0, 0.0)
    r = np.clip(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0), -0.33, 0.33)

    def mom(k):
        kk = min(k, n_days - 1)
        b = p[-kk]
        m = np.where(b > 0.0, p[-1] / b - 1.0, 0.0)
        return np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

    m5, m10, m20, m60 = mom(6), mom(11), mom(21), mom(61)
    r20 = r[-20:]
    r5 = r[-5:]
    vol = np.nan_to_num(np.std(r20, axis=0)) + 1e-4
    r1 = r20[-1]
    r5sum = np.nan_to_num(np.sum(r5, axis=0))
    held = pos > 0

    # ---------- 2. 新闻情绪 ----------
    pos_kw = ("涨", "利好", "增长", "突破", "中标", "回购", "盈利", "超预期")
    neg_kw = ("跌", "利空", "下滑", "亏损", "退市", "处罚", "减持",
              "质押", "立案", "警示", "调查", "风险")
    s_cnt, n_cnt = 0, 0
    for t in news:
        t = str(t)
        n_cnt += 1
        if any(k in t for k in pos_kw):
            s_cnt += 1
        elif any(k in t for k in neg_kw):
            s_cnt -= 1
    sent = s_cnt / n_cnt if n_cnt > 0 else 0.0

    # ---------- 3. 市场状态：严格判别，走弱/存疑一律空仓（每日检查） ----------
    mt = tradable & (vol > 1e-3)
    if mt.any():
        mkt10 = float(np.mean(m10[mt]))
        mkt20 = float(np.mean(m20[mt]))
        brd10 = float(np.mean(m10[mt] > 0.0))
        brd20 = float(np.mean(m20[mt] > 0.0))
    else:
        mkt10 = mkt20 = 0.0
        brd10 = brd20 = 0.5

    bear = (mkt10 < -0.004) or (brd10 < 0.42) or (mkt20 < -0.03) or (sent <= -0.6)
    bull = (mkt10 > 0.004) and (brd10 >= 0.50) and (mkt20 > -0.005) and (sent > -0.5)

    if bear:
        return w                                  # 熊市/风险期：立即清仓防守
    if bull:
        expo = 0.75
    elif (mkt10 > 0.0) and (brd20 >= 0.5):
        expo = 0.40
    else:
        expo = 0.0
    if expo <= 0.02:
        return w

    # ---------- 4. 个股止损：每日检查，不等调仓日 ----------
    stop = held & ((r1 < -0.06) | (r5sum < -0.09) | (m5 < -0.06))

    # ---------- 5. 调仓节奏：每5个交易日调仓一次，其余日零交易 ----------
    rebal = (n_days % 5 == 1)
    if not rebal:
        if held.any():
            w = cur_w.copy()
            if stop.any():
                w[stop] = 0.0
            return np.clip(w, 0.0, 1.0)
        return w

    # ---------- 6. 选股：长中趋势共振 + 低波动 + 不追高 ----------
    smooth = np.nan_to_num(np.mean(r20 > 0.0, axis=0))
    score = 0.6 * m20 / (20.0 * vol) + 0.4 * m60 / (60.0 * vol) + 0.15 * smooth

    ok_stock = tradable & (p[-1] >= 2.0) & np.isfinite(score)
    ok_stock &= (m20 > 0.02) & (m10 > 0.0) & (m60 > 0.0)   # 三重趋势确认
    ok_stock &= (r1 < 0.095)                               # 不追当日大涨
    ok_stock &= (r5sum < 0.14)                             # 不追短期暴涨
    ok_stock &= (m20 < 0.45)                               # 避开过热票
    ok_stock &= ~stop
    score = np.where(ok_stock, score, -np.inf)

    key = score + np.where(held, 0.3, 0.0)                 # 留仓粘性，降换手
    key[~np.isfinite(key)] = -np.inf

    K = 5
    order = np.argsort(-key)
    fin = np.isfinite(key)

    keep = [i for i in order if fin[i] and held[i]][:K]
    slots = K - len(keep)
    fresh = []
    if slots > 0:
        for i in order:
            if len(fresh) >= slots:
                break
            if fin[i] and not held[i] and score[i] > 0.25:
                fresh.append(i)
    pick = np.array(keep + fresh, dtype=int)
    if pick.size == 0:
        return w                                           # 无合格标的 => 空仓

    # ---------- 7. 等权配置，单票上限18% ----------
    wt = np.full(pick.size, expo / pick.size)
    wt = np.minimum(wt, 0.18)
    w[pick] = wt
    return np.clip(w, 0.0, 1.0)