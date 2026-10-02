import numpy as np


def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    if n_stocks == 0 or n_days < 2:
        return w

    # ---------- 1. 数据清洗 ----------
    p = np.array(prices, dtype=np.float64)
    p[~np.isfinite(p)] = 0.0
    tradable = p[-1] > 0.0
    if n_days >= 3:
        frozen = (p[-1] == p[-2]) & (p[-2] == p[-3])   # 近3日价格不动 => 疑似停牌
        tradable &= ~frozen

    prev = p[:-1]
    safe_prev = np.where(prev > 0.0, prev, 1.0)
    r = np.where(prev > 0.0, p[1:] / safe_prev - 1.0, 0.0)
    r = np.clip(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0), -0.33, 0.33)

    # ---------- 2. 因子：波动率调整动量 ----------
    def mom(k):
        kk = min(k, n_days - 1)
        b = p[-kk]
        m = np.where(b > 0.0, p[-1] / b - 1.0, 0.0)
        return np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

    mom_l = mom(21)                                    # 20日动量
    mom_s = mom(6)                                     # 5日动量(短期确认)
    r20 = r[-20:]
    vol = np.nan_to_num(np.std(r20, axis=0)) + 1e-4    # 日波动率
    up_ratio = np.nan_to_num(np.mean(r20 > 0.0, axis=0))

    score = 0.6 * mom_l / (20.0 * vol) + 0.2 * mom_s / (5.0 * vol) + 0.2 * up_ratio

    max_up = np.max(r20, axis=0)
    limit = np.where(max_up >= 0.19, 0.196,
                     np.where(max_up >= 0.095, 0.099, 0.052))
    score = np.where(r20[-1] >= limit, score - 1.0, score)   # 涨停不追
    score = np.where(p[-1] < 1.5, score - 0.5, score)        # 仙股惩罚

    # ---------- 3. 市场状态识别 + 情绪 => 熊市防守降仓 ----------
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
    sent = s_cnt / n_cnt if n_cnt > 0 else 0.0         # [-1, 1]

    mt = tradable
    mkt_mom = float(np.mean(mom_l[mt])) if mt.any() else 0.0    # 市场趋势
    breadth = float(np.mean(mom_l[mt] > 0.0)) if mt.any() else 0.5
    bear = (mkt_mom < -0.02) or (breadth < 0.40) or (sent <= -0.5)

    expo = 0.70 + 0.15 * sent + 3.0 * mkt_mom
    if bear:
        expo *= 0.45                                   # 熊市总仓位砍半
        score = np.where(mom_l <= 0.0, -np.inf, score)  # 只留强势股，否则空仓
    expo = float(np.clip(round(expo * 10.0) / 10.0, 0.10, 1.0))  # 阶梯化降换手

    # ---------- 4. 止损 + 留仓优先(降换手) ----------
    held = np.asarray(position) > 0
    stop = held & ((mom_s < -0.09) | (r20[-1] < -0.095) | (mom_l < -0.15))
    score = score + np.where(held & ~stop, 0.25, 0.0)  # 已持仓加分，减少换手
    score = np.where(held & (mom_l < -0.03) & ~stop, score - 0.3, score)
    score[~tradable] = -np.inf
    score[stop] = -np.inf
    score[~np.isfinite(score)] = -np.inf

    order = np.argsort(-score)
    fin = np.isfinite(score)
    med = float(np.median(score[fin])) if fin.any() else 0.0

    keep = [i for i in order if fin[i] and held[i] and score[i] >= med][:8]
    room = 12 - len(keep)
    fresh = [i for i in order if fin[i] and not held[i]][:min(room, 4)]
    pick = np.array(keep + fresh, dtype=int)
    if pick.size == 0:
        return w                                       # 无可买 => 空仓防守

    # ---------- 5. 逆波动率加权(单票<=15%) ----------
    iv = 1.0 / vol[pick]
    wt = iv / iv.sum() * expo
    cap = 0.15
    for _ in range(2):
        over = wt > cap
        if not over.any():
            break
        excess = float(np.sum(wt[over] - cap))
        wt[over] = cap
        rms = wt < cap
        rs = float(np.sum(wt[rms]))
        if rs > 1e-12:
            wt[rms] += excess * wt[rms] / rs
    wt = np.minimum(wt, cap)
    w[pick] = wt
    return np.clip(w, 0.0, 1.0)