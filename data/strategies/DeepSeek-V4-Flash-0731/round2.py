import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    
    # 需要至少21天数据来计算动量与波动率
    if n_days < 21:
        return w
    
    # 获取关键价格（避免越界：使用切片尾部）
    last_price = prices[-1, :]
    price_5_ago = prices[-5, :] if n_days >= 5 else prices[0, :]
    price_20_ago = prices[-20, :] if n_days >= 20 else prices[0, :]
    price_60_ago = prices[-60, :] if n_days >= 60 else prices[0, :]
    
    # 收益率（统一使用安全除法）
    eps = 1e-8
    ret_60 = last_price / (price_60_ago + eps) - 1.0
    ret_20 = last_price / (price_20_ago + eps) - 1.0
    ret_5 = last_price / (price_5_ago + eps) - 1.0
    
    # 过去20天日收益的波动率（使用最后20个价格点，差分需对齐）
    lookback = min(20, n_days - 1)  # 确保diff有足够长度
    if n_days >= 21:
        price_window = prices[-21:-1, :]  # 21个点产生20个收益
        daily_ret = np.diff(price_window, axis=0) / (price_window[:-1, :] + eps)
        vol = np.std(daily_ret, axis=0)
    else:
        vol = np.full(n_stocks, eps)
    
    # 有效性过滤：价格为正，有足够历史，波动率>0
    valid = (last_price > 0) & (price_20_ago > 0) & (price_5_ago > 0) & (vol > eps)
    
    # 动量得分：动量/波动率，突出稳定性
    score = np.zeros(n_stocks)
    if np.any(valid):
        score[valid] = (ret_60[valid] * 0.4 + ret_20[valid] * 0.35 + ret_5[valid] * 0.25) / (vol[valid] + eps)
    
    # 选择得分最高的股票，但排除价格=0或无效的
    valid_indices = np.where(valid)[0]
    if len(valid_indices) == 0:
        return w
    
    # 对有效股票排序，取前20名（若不足20则全选）
    sorted_valid = valid_indices[np.argsort(score[valid_indices])[::-1]]
    n_select = min(20, len(sorted_valid))
    selected = sorted_valid[:n_select]
    
    # 等权权重，单只权重上限20%
    if n_select > 0:
        weight = 1.0 / n_select
        if weight > 0.2:
            # 最多选5只（1/5=0.2），但这里仍按20只，权重0.05<0.2，不会触发
            pass
        w[selected] = weight
    
    # 保证权重总和<=1，若超过则缩放
    total_w = np.sum(w)
    if total_w > 1.0 + 1e-12:
        w = w / total_w
    
    # 强制单只权重 <= 0.2（保险使用）
    max_w = np.max(w)
    if max_w > 0.2 + 1e-12:
        # 将超限股票压缩到0.2
        w[w > 0.2] = 0.2
        total = np.sum(w)
        if total > 1.0 + 1e-12:
            w = w / total
    
    # 非负修正（确保数值稳定）
    w = np.maximum(w, 0.0)
    # 归一化（若总和>0）
    total = np.sum(w)
    if total > 1e-12:
        w = w / total
    
    return w