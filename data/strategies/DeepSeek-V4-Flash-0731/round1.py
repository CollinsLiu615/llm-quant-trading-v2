import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    
    # 计算收益率（使用收盘价，最后一天与之前比较）
    if n_days < 2:
        return w
    
    # 过去60天收益率（累计），用最后一天/第一天 - 1
    ret_60 = np.divide(prices[-1], prices[0] + 1e-12, out=np.ones(n_stocks), where=prices[0]!=0) - 1.0
    
    # 过去5天收益率
    ret_5 = np.divide(prices[-1], prices[-5] + 1e-12, out=np.ones(n_stocks), where=prices[-5]!=0) - 1.0
    
    # 过去20天收益率
    ret_20 = np.divide(prices[-1], prices[-20] + 1e-12, out=np.ones(n_stocks), where=prices[-20]!=0) - 1.0
    
    # 波动率（过去20天日收益率的std）
    daily_ret = np.diff(prices[-20:], axis=0) / (prices[-21:-1] + 1e-12)
    vol = np.std(daily_ret, axis=0)
    
    # 过滤：价格>0，有足够历史，非停牌（最后一天价格>0）
    valid = (prices[-1] > 0) & (prices[-20] > 0) & (prices[0] > 0) & (vol > 0)
    
    # 动量综合得分：60日+20日+5日，权重不同
    score = np.zeros(n_stocks)
    score[valid] = (ret_60[valid] * 0.4 + ret_20[valid] * 0.35 + ret_5[valid] * 0.25) / (vol[valid] + 1e-8)
    
    # 剔除当前持仓中的停牌股（如果有），但保留持仓。
    # 简单起见：不剔除持仓，持仓继续持有，但权重限制。
    
    # 排名选股
    top_idx = np.argsort(score)[::-1][:20]  # 得分最高的20只
    
    # 将已持仓的股票也考虑在内，优先保留，但这里简化直接选高分
    selected = top_idx
    
    # 分配权重：等权，但不超过20%
    weight_per_stock = 1.0 / len(selected) if len(selected)>0 else 0.0
    if weight_per_stock > 0.2:
        weight_per_stock = 0.2
        # 如果权重超过20%，则最多选5只？但这里允许20只，等权不超过20%可行（因为1/5=0.2），所以我们最多选5只。
        # 但目标是20只，既然等权0.05<0.2，没问题，我们只限制单只<=0.2，所以等权完全满足。
    w[selected] = weight_per_stock
    
    # 确保 sum<=1，并调整权重，如果持仓权重过大，则按持仓比例减少
    total_w = np.sum(w)
    if total_w > 1.0:
        w = w / total_w
    
    # 可选：对持仓股票（position>0）额外倾斜，但为简单，保持原权重
    
    # 检查单只权重不超过0.2，如果不满足则压缩
    max_w = np.max(w)
    if max_w > 0.2:
        # 将超过0.2的股票权重缩到0.2，并再均衡
        w[w>0.2] = 0.2
        total = np.sum(w)
        if total > 1.0:
            w = w / total
    
    # 最终确保非负
    w = np.maximum(w, 0)
    
    return w