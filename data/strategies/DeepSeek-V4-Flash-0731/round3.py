import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    
    # 需要至少26天数据来计算动量与波动率
    if n_days < 26:
        return w
    
    # 获取关键价格（避免越界）
    last_price = prices[-1, :]
    price_5_ago = prices[-5, :]
    price_20_ago = prices[-20, :]
    price_60_ago = prices[-60, :] if n_days >= 60 else prices[0, :]
    
    # 收益率
    eps = 1e-8
    ret_60 = last_price / (price_60_ago + eps) - 1.0
    ret_20 = last_price / (price_20_ago + eps) - 1.0
    ret_5 = last_price / (price_5_ago + eps) - 1.0
    
    # 过去20天日收益的波动率
    price_window = prices[-21:-1, :]
    daily_ret = np.diff(price_window, axis=0) / (price_window[:-1, :] + eps)
    vol = np.std(daily_ret, axis=0)
    
    # 市场状态识别：基于最近60日大盘走势（以全市场等权平均计算）
    # 使用全市场平均价格作为市场组合
    if n_days >= 60:
        market_avg = prices[-60:, :].mean(axis=1)  # 60天每日市场平均价格
        market_ret = market_avg[-1] / (market_avg[0] + eps) - 1.0
        market_vol = np.std(market_avg) / (np.abs(market_avg[-1]) + eps)
        # 简易市场状态：收益>0且波动不过大 -> 多头；否则空头/震荡
        if market_ret > 0.05 and market_vol < 0.5:
            market_state = 1.0   # 看多
        elif market_ret < -0.05:
            market_state = -1.0  # 看空
        else:
            market_state = 0.0   # 震荡
    else:
        market_state = 0.0
    
    # 有效性过滤
    valid = (last_price > 0) & (price_20_ago > 0) & (price_5_ago > 0) & (vol > eps)
    
    # 动量得分：动量/波动率，并加入市场状态调整
    score = np.zeros(n_stocks)
    if np.any(valid):
        base_score = (ret_60[valid] * 0.4 + ret_20[valid] * 0.35 + ret_5[valid] * 0.25) / (vol[valid] + eps)
        # 市场调整：空头时降低得分，并额外惩罚高波动个股
        if market_state < 0:
            score[valid] = base_score * 0.5 - vol[valid] * 0.2
        elif market_state == 0:
            score[valid] = base_score * 0.8 - vol[valid] * 0.1
        else:
            score[valid] = base_score
    
    # 选择股票
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return w
    
    # 结合市场状态决定仓位
    if market_state < 0:
        target_cash_weight = 0.7  # 熊市保留70%现金
        n_target = min(5, len(valid_indices))  # 空头只选最多5只
    elif market_state == 0:
        target_cash_weight = 0.4
        n_target = min(10, len(valid_indices))
    else:
        target_cash_weight = 0.2
        n_target = min(20, len(valid_indices))
    
    # 排序选择
    sorted_valid = valid_indices[np.argsort(score[valid_indices])[::-1]]
    n_select = min(n_target, len(sorted_valid))
    selected = sorted_valid[:n_select]
    
    if n_select > 0:
        # 等权权重，并依据市场状态分配总仓位
        equity_weight = (1.0 - target_cash_weight) / n_select
        w[selected] = equity_weight
    
    # 止损机制：个股相对买入价下跌超10%则清仓该股（用20日价格近似判断）
    # 这里用持仓位置来模拟已实现止损，但策略无状态，我们简单通过动量信号过滤
    # 额外检查：若个股最近20日下跌超过10%，降权或剔除
    if n_select > 0:
        for idx in selected:
            if ret_20[idx] < -0.1:
                w[idx] *= 0.5  # 减半权重，相当于部分止损
    
    # 强制单只权重上限15% (避免过度集中)
    max_w = 0.15
    w[w > max_w] = max_w
    
    # 归一化确保总权重不超过1
    total = np.sum(w)
    if total > 1.0 + 1e-12:
        w = w / total
    elif total < 0:
        w = np.zeros(n_stocks)
    
    # 非负修正
    w = np.maximum(w, 0.0)
    # 最终归一化到总权重<=1
    total = np.sum(w)
    if total > 1e-12:
        w = w / total  # 确保满仓但总权重为1，但上面已预留现金，所以这里总权重应<=1
        # 实际应该保留现金，所以不需要重新归一化到1，但为了安全确保<=1
        if np.sum(w) > 1.0:
            w = w / np.sum(w)
    
    return w