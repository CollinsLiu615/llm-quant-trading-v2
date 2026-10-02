def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    # 清洗价格
    p_last = prices[-1]
    p_5 = prices[-6] if n_days >= 6 else p_last
    p_20 = prices[-21] if n_days >= 21 else prices[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        ret_5 = p_last / p_5 - 1
        ret_20 = p_last / p_20 - 1
        rets = prices[1:] / prices[:-1] - 1
    ret_5[~np.isfinite(ret_5)] = np.nan
    ret_20[~np.isfinite(ret_20)] = np.nan
    ret_5[p_5 <= 0] = np.nan
    ret_20[p_20 <= 0] = np.nan
    rets[~np.isfinite(rets)] = np.nan
    rets[rets > 10] = np.nan
    rets[rets < -0.9] = np.nan
    # 波动率
    if n_days >= 21:
        vol20 = np.nanstd(rets[-20:], axis=0)
        max_abs = np.nanmax(np.abs(rets), axis=0)
    else:
        vol20 = np.nanstd(rets, axis=0)
        max_abs = np.nanmax(np.abs(rets), axis=0)
    # 有效掩码
    valid = np.isfinite(p_last) & (p_last > 0) & np.isfinite(ret_5) & np.isfinite(ret_20) & np.isfinite(vol20)
    valid = valid & (vol20 > 0.001) & (vol20 < 0.08) & (p_last > 1.0)
    # 排除疑似ST/停牌：长期最大涨幅<5.7%且波动极小
    st_mask = np.isfinite(max_abs) & (max_abs < 0.057) & (vol20 < 0.015)
    valid[st_mask] = False
    # 排除一字板/极端
    valid[np.abs(ret_5) > 0.6] = False
    if np.sum(valid) < 5:
        return w
    # 新闻情绪
    sentiment = 0
    try:
        if news is not None and len(news) > 0:
            text = " ".join([str(x) for x in news if isinstance(x, str)])
            pos_words = ["利好", "上涨", "增长", "突破", "新高", "回暖", "向好", "牛市", "增持", "买入"]
            neg_words = ["利空", "下跌", "暴跌", "风险", "监管", "处罚", "暴雷", "退市", "减持", "熊市", "大跌", "下行"]
            pos = 0
            neg = 0
            for wd in pos_words:
                pos += text.count(wd)
            for wd in neg_words:
                neg += text.count(wd)
            sentiment = pos - neg
    except:
        sentiment = 0
    # 市场趋势
    m_ret = np.nanmean(ret_20[valid])
    if not np.isfinite(m_ret):
        m_ret = 0
    # 评分：短期反转 + 低波 + 长期动量微调
    score = np.full(n_stocks, -np.inf)
    # 风险调整反转
    raw = -ret_5 / (vol20 + 0.02) + 0.15 * ret_20 / (vol20 + 0.02)
    # 去极值
    v = raw[valid]
    if v.size > 0:
        lo = np.nanpercentile(v, 1)
        hi = np.nanpercentile(v, 99)
        raw = np.clip(raw, lo, hi)
    score[valid] = raw[valid]
    # 持仓粘性，降低换手，缓解T+1
    try:
        held = position > 0
        if np.any(held & valid):
            std_s = np.nanstd(score[valid])
            if np.isfinite(std_s) and std_s > 0:
                score[held & valid] += 0.12 * std_s
    except:
        pass
    # 选择
    order = np.argsort(score)
    # 取有效的前20
    selected = []
    for idx in order[::-1]:
        if valid[idx] and np.isfinite(score[idx]):
            selected.append(idx)
            if len(selected) >= 20:
                break
    if len(selected) == 0:
        return w
    # 总仓位由情绪和市场决定
    total = 0.95
    if sentiment < -2:
        total = 0.5
    elif sentiment < 0:
        total = 0.7
    if m_ret < -0.08:
        total *= 0.6
    elif m_ret < -0.03:
        total *= 0.8
    # 波动率高降低仓位
    mean_vol = np.nanmean(vol20[valid])
    if np.isfinite(mean_vol) and mean_vol > 0.035:
        total *= 0.8
    total = float(np.clip(total, 0.3, 0.95))
    per = total / len(selected)
    if per > 0.2:
        per = 0.2
    for idx in selected:
        w[idx] = per
    # 归一化保证sum<=1
    s = np.sum(w)
    if s > 1:
        w = w / s
    return w