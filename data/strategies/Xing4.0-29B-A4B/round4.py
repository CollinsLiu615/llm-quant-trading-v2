import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)

    if n_days < 6:
        return w

    # === 1. 市场状态识别 ===
    log_ret = np.diff(np.log(prices))
    vol = np.std(log_ret, axis=0)
    med_vol = np.median(vol[-20:]) if n_days >= 21 else vol[-1]

    ma5 = np.mean(prices[-5:, :], axis=0)
    ma10 = np.mean(prices[-10:, :], axis=0) if n_days >= 11 else ma5

    is_hv = vol[-1] > med_vol
    is_lv = vol[-1] < med_vol
    is_up = ma5[-1] > ma5[-2]
    is_dn = ma5[-1] < ma5[-2]

    bear_mask = is_hv & is_dn
    bull_mask = is_lv & is_up
    neutral_mask = ~(bear_mask | bull_mask)

    market_factor = np.ones(n_stocks)
    market_factor[bear_mask] = 0.15
    market_factor[bull_mask] = 1.0
    market_factor[neutral_mask] = 0.5

    # === 2. 收益率计算 ===
    ret = prices[-1, :] / prices[-5, :] - 1.0

    # === 3. 新闻情绪评分 ===
    score = np.zeros(n_stocks)
    pos_kw = ["利好", "上涨", "突破", "增长", "盈利", "超预期", "大涨"]
    neg_kw = ["利空", "下跌", "破位", "亏损", "下滑", "暴跌", "风险"]

    if isinstance(news, list) and len(news) > 0 and isinstance(news[0], list):
        for i in range(n_days):
            for j in range(n_stocks):
                if ret[j] > 0:
                    text = news[i][j] if j < len(news[i]) else ""
                    if any(k in text for k in pos_kw):
                        score[j] += 1
                    elif any(k in text for k in neg_kw):
                        score[j] -= 1

    # === 4. 信号过滤 ===
    signal = score * ret
    valid = signal > 0

    if not np.any(valid):
        return w

    # === 5. 自适应权重分配 ===
    max_w = 0.12
    remaining = 1.0

    sorted_idx = np.argsort(signal)[::-1]
    for idx in sorted_idx:
        if not valid[idx] or remaining <= 0:
            continue
        norm_sig = signal[idx] / np.max(signal)
        w[idx] = min(max_w * norm_sig, remaining)
        remaining -= w[idx]

    # 应用市场状态因子
    w *= market_factor

    # === 6. 止损机制 ===
    sl_ratio = 0.05
    for j in range(n_stocks):
        if w[j] > 0:
            buy_p = prices[-5, :][j] if j < len(prices[-5, :]) else prices[-1, :][j]
            if buy_p > 0:
                sl_line = buy_p * (1 - sl_ratio)
                if prices[-1, :][j] < sl_line:
                    w[j] = 0

    # === 7. 归一化 ===
    total = np.sum(w)
    if total > 1.0:
        w /= total

    return w