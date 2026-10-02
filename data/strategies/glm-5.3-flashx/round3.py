import numpy as np


def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    if n_stocks == 0 or n_days < 25:
        return w

    # ---------- 1. 数据清洗 ----------
    p = np.array(prices, dtype=np.float64)
    p[~np.isfinite(p)] = 0.0
    tradable = p[-1] > 0.0
    if n_days >= 3:
        frozen = (p[-1] == p[-2]) & (p[-2] == p[-3])   # 近3日价格不动 => 疑似停牌
        tradable &= ~frozen

    prev = p[:-1]
    ok = prev > 0.0
    r = np.where(ok, p[1:] / np.where(ok, prev, 1.0) - 1.0, 0.0)
    r = np.clip(np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0), -0.33, 0.33)

    # ---------- 2. 因子：中短双确认的稳健趋势 ----------
    def mom(k):
        kk = min(k, n_days - 1)
        b = p[-kk]
        m = np.where(b > 0.0, p[-1] / b - 1.0, 0.0)
        return np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)

    mom_l = mom(21)                                    # 20日动量
    mom_s = mom(6)                                     # 5日动量
    m10 = mom(11)                                      # 10日动量
    r20 = r[-20:]
    vol = np.nan_to_num(np.std(r20, axis=0)) + 1e-4    # 日波动率
    smooth = np.nan_to_num(np.mean(r20 > 0.0, axis=0))  # 上涨天数占比(平滑度)

    score = 0.5 * mom_l / (20.0 * vol) + 0.3 * m10 / (10.0 * vol) + 0.2 * smooth
    score = np.where((mom_l <= 0.0) | (mom_s <= 0.0), -np.inf, score)  # 双趋势须为正

    # 不追高：涨停不买，短期暴涨(易回吐)重罚
    r5max = np.max(r[-5:], axis=0)
    limit = np.where(r5max >= 0.19, 0.196,
                     np.where(r5max >= 0.095, 0.099, 0.052))
    score = np.where(r20[-1] >= limit, -np.inf, score)
    score = np.where(r5max >= 0.15, score - 1.0, score)
    score = np.where(p[-1] < 2.0, score - 0.5, score)  # 仙股惩罚

    # ---------- 3. 新闻情绪 ----------
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

    # ---------- 4. 市场状态：趋势+双周期宽度+情绪，熊市坚决降仓 ----------
    mt = tradable & (vol > 1e-3)
    if mt.any():
        mkt_l = float(np.mean(mom_l[mt]))
        mkt_s = float(np.mean(mom_s[mt]))
        brd_l = float(np.mean(mom_l[mt] > 0.0))
        brd_s = float(np.mean(mom_s[mt] > 0.0))
    else:
        mkt_l = mkt_s = 0.0
        brd_l = brd_s = 0.5
    bull = (mkt_l > 0.005) and (brd_l >= 0.45) and (brd_s >= 0.45) and (sent > -0.4)
    bear = (mkt_s < -0.01) or (brd_s < 0.35) or (sent <= -0.6)

    expo = 0.65 + 0.10 * sent + 2.0 * min(mkt_l, 0.1)
    if bear:
        expo *= 0.30                                   # 熊市仅留三成仓
    elif not bull:
        expo *= 0.60                                   # 震荡市半仓防守
    expo = float(np.clip(round(expo * 10.0) / 10.0, 0.0, 0.9))  # 阶梯化降换手
    if expo < 0.05:
        return w                                       # 极弱市空仓

    # ---------- 5. 持仓止损：动量转弱/单日大跌立即退出 ----------
    held = np.asarray(position) > 0
    stop = held & ((r20[-1] < -0.05) | (mom_s < -0.05) | (mom_l < -0.08))

    # ---------- 6. 选股：留仓优先+高门槛，每日最多新增2只 => 大幅降换手 ----------
    key = np.where(np.isfinite(score), score, -np.inf)
    key[stop] = -np.inf
    key = key + np.where(held, 0.6, 0.0)               # 已持仓粘性加分
    key[~tradable] = -np.inf
    key[~np.isfinite(key)] = -np.inf

    order = np.argsort(-key)
    fin = np.isfinite(key)
    K = 8
    keep = [i for i in order if fin[i] and held[i]][:K]
    # 新增须原始分数过硬，且只补空槽、每日至多2只
    fresh = [i for i in order
             if fin[i] and not held[i] and score[i] > 0.25]
    fresh = fresh[:max(0, min(2, K - len(keep)))]
    pick = np.array(keep + fresh, dtype=int)
    if pick.size == 0:
        return w                                       # 无合格标的 => 空仓防守

    # ---------- 7. 等权配置(权重稳定=>低换手)，单票<=15% ----------
    wt = np.full(pick.size, expo / pick.size)
    wt = np.minimum(wt, 0.15)                          # 超额部分留作现金
    w[pick] = wt
    return np.clip(w, 0.0, 1.0)