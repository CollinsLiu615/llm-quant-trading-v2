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
    safe_prev = np.where(prev > 0.0, prev, np.nan)
    r = p[1:] / safe_prev - 1.0                        # 日收益率 (n_days-1, n_stocks)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    r = np.clip(r, -0.33, 0.33)

    # ---------- 2. 因子：波动率调整动量 ----------
    def ref(k):
        kk = min(k, n_days - 1)
        return np.where(p[-kk] > 0.0, p[-kk], np.nan)

    mom_l = np.nan_to_num(p[-1] / ref(20) - 1.0)       # 20日动量
    mom_s = np.nan_to_num(p[-1] / ref(5) - 1.0)        # 5日动量(短期确认)
    vol = np.nan_to_num(np.std(r, axis=0)) + 1e-4      # 日波动率
    up_ratio = np.nan_to_num(np.mean(r > 0.0, axis=0)) # 上涨天数占比

    score = 0.6 * mom_l / (20.0 * vol) + 0.2 * mom_s / (5.0 * vol) + 0.2 * up_ratio

    # 涨停难以买入：按该股历史最大涨幅推断板块涨跌停幅度并惩罚一字板
    max_up = np.max(r, axis=0)
    limit = np.where(max_up >= 0.19, 0.196,
                     np.where(max_up >= 0.095, 0.099, 0.052))
    at_limit = r[-1] >= limit
    score = np.where(at_limit, score - 1.0, score)

    # 低价/仙股惩罚；已持仓小幅加分(降低T+1下的无谓换手)
    score = np.where(p[-1] < 1.5, score - 0.5, score)
    score = score + np.where(position > 0, 0.12, 0.0)
    score[~tradable] = -np.inf

    # ---------- 3. 新闻情绪 + 市场环境 => 目标总仓位 ----------
    pos_kw = ("涨", "利好", "增长", "突破", "中标", "回购", "盈利", "超预期", "回暖")
    neg_kw = ("跌", "利空", "下滑", "亏损", "退市", "处罚", "减持",
              "质押", "立案", "警示", "风险", "调查")
    s, n = 0, 0
    for t in news:
        try:
            t = str(t)
        except Exception:
            continue
        n += 1
        if any(k in t for k in pos_kw):
            s += 1
        elif any(k in t for k in neg_kw):
            s -= 1
    sent = s / n if n > 0 else 0.0                     # [-1, 1]
    mkt = float(np.mean(mom_s[tradable])) if tradable.any() else 0.0
    expo = 0.80 + 0.20 * sent + 2.0 * mkt              # 情绪与市场趋势共同调节总仓位
    expo = max(0.30, min(1.0, expo))

    # ---------- 4. 选Top20 + 逆波动率加权(单票<=15%) ----------
    order = np.argsort(-score)                         # score降序
    pick = order[:20]
    pick = pick[np.isfinite(score[pick])]
    if pick.size == 0:
        return w

    iv = 1.0 / vol[pick]
    wt = iv / iv.sum() * expo
    cap = 0.15
    for _ in range(2):                                 # 超额权重再分配(两轮)
        over = wt > cap
        if not over.any():
            break
        excess = float(np.sum(wt[over] - cap))
        wt[over] = cap
        room = wt < cap
        rs = float(np.sum(wt[room]))
        if rs > 1e-12:
            wt[room] += excess * wt[room] / rs
    wt = np.minimum(wt, cap)
    w[pick] = wt

    tot = float(np.sum(w))
    if tot > 1.0:
        w = w * (1.0 / tot)
    return np.clip(w, 0.0, 1.0)