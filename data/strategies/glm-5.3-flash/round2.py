import numpy as np


def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)

    # ---------- 1. 数据清洗 ----------
    p = np.array(prices, dtype=float, copy=True)
    p[~np.isfinite(p)] = np.nan
    p[p <= 0] = np.nan

    with np.errstate(invalid="ignore", divide="ignore"):
        last = p[-1]
        prev = p[-2] if n_days >= 2 else last
        ret1 = last / prev - 1.0
        ret5 = last / p[max(0, n_days - 6)] - 1.0
        m20 = last / p[max(0, n_days - 21)] - 1.0
        m60 = last / p[0] - 1.0
        dr = (p[1:] / p[:-1] - 1.0) if n_days >= 2 else np.zeros((1, n_stocks))
        vol = np.nanstd(dr[-30:], axis=0)

    # ---------- 2. 新闻情绪 + 市场环境 -> 总仓位 exposure ----------
    bull_kw = ("利好", "预增", "增长", "中标", "回购", "增持", "上调", "突破",
               "签约", "获批", "分红", "盈利", "订单", "合作", "净流入")
    bear_kw = ("利空", "预亏", "亏损", "减持", "处罚", "立案", "调查", "退市",
               "质押", "违约", "诉讼", "冻结", "下滑", "终止", "警示", "跌停")
    nb = 0
    ns = 0
    for t in news:
        if not t:
            continue
        nb += sum(1 for kw in bull_kw if kw in t)
        ns += sum(1 for kw in bear_kw if kw in t)
    sent = (nb - ns) / float(nb + ns + 2.0)

    tradable = np.isfinite(last) & np.isfinite(m20) & np.isfinite(vol) & (last >= 1.5)
    regime = float(np.nanmean(m20[tradable])) if tradable.any() else 0.0
    exposure = 0.55 + 0.25 * np.clip(regime * 5.0, -1.0, 1.0) + 0.15 * np.clip(sent, -1.0, 1.0)
    exposure = float(np.clip(exposure, 0.25, 0.92))

    # ---------- 3. 因子打分: 中期动量 + 长期动量 - 低波动 - 短期反转 ----------
    def zscore(x):
        m = np.nanmean(x)
        s = np.nanstd(x)
        if (not np.isfinite(s)) or s < 1e-12:
            return np.zeros(n_stocks)
        return np.clip((x - m) / s, -3.0, 3.0)

    score = 1.0 * zscore(m20) + 0.6 * zscore(m60) - 0.7 * zscore(vol) - 0.5 * zscore(ret5)
    score = np.where(np.isfinite(score), score, -1e9)

    # 已持有股票小幅加分, 降低换手成本(佣金+印花税)
    held = (np.asarray(position) > 0) & tradable
    score = score + held * 0.30

    # ---------- 4. 可交易过滤: 剔除疑似涨跌停(T+1 下追涨/接飞刀风险) ----------
    ret1f = np.nan_to_num(ret1, nan=0.0, posinf=0.5, neginf=-0.5)
    ok = tradable & (ret1f <= 0.095) & (ret1f >= -0.095)
    idx = np.where(ok)[0]
    if idx.size == 0:
        return w

    # ---------- 5. 选前 20 只, 反波动率加权, 单票上限 20% ----------
    k = int(min(20, idx.size))
    s_ok = score[idx]
    top = np.argpartition(-s_ok, k - 1)[:k]
    sel = idx[top]
    iv = 1.0 / (vol[sel] + 1e-4)
    wsel = iv / iv.sum() * exposure
    w[sel] = np.minimum(wsel, 0.20)

    # ---------- 6. 安全检查: 无 nan/inf, 范围 [0, 0.2], sum <= 1 ----------
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.clip(w, 0.0, 0.20)
    if w.sum() > 1.0:
        w = w / w.sum()
    return w