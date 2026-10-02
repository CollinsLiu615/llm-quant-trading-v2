import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    
    # 数据不足时返回空仓
    if n_days < 21:
        return w
    
    last_price = prices[-1, :]
    eps = 1e-8
    
    # 计算各周期收益率（使用可用数据）
    price_5_ago = prices[-5, :] if n_days >= 5 else prices[0, :]
    price_10_ago = prices[-10, :] if n_days >= 10 else prices[0, :]
    price_20_ago = prices[-20, :] if n_days >= 20 else prices[0, :]
    price_60_ago = prices[-60, :] if n_days >= 60 else prices[0, :]
    
    ret_60 = last_price / (price_60_ago + eps) - 1.0
    ret_20 = last_price / (price_20_ago + eps) - 1.0
    ret_10 = last_price / (price_10_ago + eps) - 1.0
    ret_5 = last_price / (price_5_ago + eps) - 1.0
    
    # 波动率（过去21天日收益）
    price_window = prices[-22:-1, :] if n_days >= 22 else prices[:-1, :]
    if price_window.shape[0] >= 2:
        daily_ret = np.diff(price_window, axis=0) / (price_window[:-1, :] + eps)
        vol = np.std(daily_ret, axis=0)
    else:
        vol = np.ones(n_stocks) * 0.02
    
    vol = np.maximum(vol, 1e-4)
    
    # 市场状态：基于20日市场平均收益
    if n_days >= 20:
        market_avg = prices[-20:, :].mean(axis=1)
        market_ret = market_avg[-1] / (market_avg[0] + eps) - 1.0
        market_vol = np.std(market_avg) / (np.abs(market_avg[-1]) + eps)
        
        if market_ret > 0.02 and market_vol < 0.3:
            market_state = 1
        elif market_ret < -0.02:
            market_state = -1
        else:
            market_state = 0
    else:
        market_state = 0
    
    # 有效股票过滤
    valid = (last_price > 0) & (price_20_ago > 0) & (vol > 1e-4)
    
    # 动量得分（基于20日和60日，按波动率调整）
    score = np.zeros(n_stocks)
    if np.any(valid):
        base_score = (ret_60 + ret_20) / (vol + eps)
        # 市场调整
        if market_state < 0:
            score[valid] = base_score * 0.3 - vol[valid] * 0.5
        elif market_state == 0:
            score[valid] = base_score * 0.6 - vol[valid] * 0.2
        else:
            score[valid] = base_score
    
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return w
    
    # 选择股票数量与仓位（熊市大幅降仓）
    if market_state < 0:
        cash_w = 0.8
        n_target = 3
    elif market_state == 0:
        cash_w = 0.5
        n_target = 8
    else:
        cash_w = 0.25
        n_target = 15
    
    sorted_valid = valid_indices[np.argsort(score[valid_indices])[::-1]]
    n_select = min(n_target, len(sorted_valid))
    selected = sorted_valid[:n_select]
    
    if n_select > 0:
        equity_w = (1.0 - cash_w) / n_select
        w[selected] = equity_w
    
    # 止损：20日跌幅大于15%的股票剔除
    if n_select > 0:
        for idx in selected:
            if ret_20[idx] < -0.15:
                w[idx] = 0.0
    
    # 单只权重上限
    w = np.minimum(w, 0.12)
    
    # 确保总权重不超过1，保留现金
    total = np.sum(w)
    if total > 1.0:
        w = w / total
    
    # 非负
    w = np.maximum(w, 0.0)
    
    return w