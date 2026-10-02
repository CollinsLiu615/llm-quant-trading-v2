def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks, dtype=float)
    try:
        # news sentiment
        sentiment = 0
        try:
            txt = " ".join(news) if news is not None else ""
            pos_words = ["利好", "增长", "上升", "突破", "新高", "回暖", "复苏", "超预期", "大涨", "向好"]
            neg_words = ["利空", "下跌", "暴跌", "风险", "下滑", "衰退", "暴雷", "跌停", "危机", "大跌"]
            for wd in pos_words:
                if wd in txt:
                    sentiment += 1
            for wd in neg_words:
                if wd in txt:
                    sentiment -= 1
        except:
            sentiment = 0

        last = prices[-1]
        # valid mask
        valid = np.isfinite(last) & (last > 0) & (last < 1e6)
        cnt = np.sum(np.isfinite(prices), axis=0)
        valid &= cnt >= 40
        # price range not flat
        with np.errstate(invalid='ignore'):
            pmax = np.nanmax(prices, axis=0)
            pmin = np.nanmin(prices, axis=0)
        prange = pmax - pmin
        valid &= np.isfinite(prange) & (prange > 1e-6)
        # last 5 days at least 4 finite
        valid &= np.sum(np.isfinite(prices[-5:]), axis=0) >= 4

        # returns and volatility
        with np.errstate(divide='ignore', invalid='ignore'):
            rets = prices[1:] / prices[:-1] - 1
        rets[~np.isfinite(rets)] = np.nan
        # filter limit moves and bad data
        rets[(rets > 0.35) | (rets < -0.35)] = np.nan

        vol20 = np.nanstd(rets[-20:], axis=0)
        # momentum
        p20 = prices[-21] if n_days >= 21 else prices[0]
        p5 = prices[-6] if n_days >= 6 else prices[0]
        mom20 = np.full(n_stocks, np.nan)
        mom5 = np.full(n_stocks, np.nan)
        m20_ok = np.isfinite(p20) & (p20 > 0) & np.isfinite(last)
        m5_ok = np.isfinite(p5) & (p5 > 0) & np.isfinite(last)
        mom20[m20_ok] = last[m20_ok] / p20[m20_ok] - 1
        mom5[m5_ok] = last[m5_ok] / p5[m5_ok] - 1

        ma20 = np.nanmean(prices[-20:], axis=0)
        dist = np.full(n_stocks, np.nan)
        ma_ok = np.isfinite(ma20) & (ma20 > 0) & np.isfinite(last)
        dist[ma_ok] = last[ma_ok] / ma20[ma_ok] - 1

        # score: risk adjusted momentum with mean reversion penalty
        score = np.full(n_stocks, -1e9, dtype=float)
        vol_ok = np.isfinite(vol20) & np.isfinite(mom20) & np.isfinite(dist)
        # base score
        base = np.zeros(n_stocks)
        base[vol_ok] = mom20[vol_ok] / (vol20[vol_ok] + 0.02)
        # penalty for over extension
        penalty = np.zeros(n_stocks)
        penalty[vol_ok] = 1.5 * np.maximum(dist[vol_ok] - 0.10, 0) + 1.0 * np.maximum(-dist[vol_ok] - 0.05, 0)
        s = base - penalty
        # apply filters
        keep = valid & vol_ok
        keep &= (vol20 > 0.005) & (vol20 < 0.045)
        keep &= (mom20 > 0.02) & (mom20 < 0.45)
        keep &= (mom5 > -0.06) & (mom5 < 0.30)
        keep &= (dist > -0.08) & (dist < 0.18)
        # turnover reduction: boost held positions
        try:
            held = position > 0
            s[held & np.isfinite(s)] += 0.015
            # keep held a bit more lenient
            keep_held = valid & vol_ok & held & (vol20 < 0.06) & (mom20 > 0.0)
            keep = keep | keep_held
        except:
            pass
        score[keep] = s[keep]
        # if too few, relax
        if np.sum(keep) < 8:
            keep2 = valid & vol_ok & (vol20 > 0.003) & (vol20 < 0.06) & (mom20 > 0.0) & (mom20 < 0.6)
            score[keep2 & ~keep] = s[keep2 & ~keep] - 0.05
            keep = keep | keep2
        # market regime
        market_mom = np.nanmean(mom20[valid]) if np.sum(valid) > 100 else 0
        # total exposure
        tot = 0.95
        if sentiment <= -2:
            tot = 0.35
        elif sentiment == -1:
            tot = 0.55
        elif market_mom < -0.02:
            tot = 0.45
        elif market_mom < 0:
            tot = 0.65
        if sentiment >= 2 and market_mom > 0.02:
            tot = 0.95

        # select top 20
        # argsort descending
        order = np.argsort(-score)
        # filter only keep with score > -1e8
        selected = []
        for idx in order:
            if score[idx] > -1e8:
                selected.append(idx)
            if len(selected) >= 20:
                break
        if len(selected) == 0:
            return w
        # weight allocation: score weighted but capped at 0.2
        sc = score[selected]
        # shift to positive
        sc_min = np.min(sc)
        sc_adj = sc - sc_min + 0.01
        # if sentiment negative, equal weight defensive
        if sentiment < 0:
            sc_adj = np.ones_like(sc_adj)
        w_sel = sc_adj / np.sum(sc_adj) * tot
        # cap 0.2
        w_sel = np.minimum(w_sel, 0.2)
        # renormalize if capped
        ssum = np.sum(w_sel)
        if ssum > tot:
            w_sel = w_sel / ssum * tot
        # if still sum < tot due to cap, distribute remainder equally
        for i, idx in enumerate(selected):
            w[idx] = w_sel[i]
        # ensure sum <=1 and each <=0.2
        if np.sum(w) > 1:
            w = w / np.sum(w) * 0.95
        return w
    except:
        return np.zeros(n_stocks, dtype=float)