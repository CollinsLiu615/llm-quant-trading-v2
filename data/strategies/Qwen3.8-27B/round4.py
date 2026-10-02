import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    
    if n_days < 10:
        return w

    # 1. 市场状态识别
    lb_market = min(20, n_days - 1)
    if lb_market <= 1:
        return w
    
    # 使用所有股票的平均近期表现作为市场代理
    market_recent = prices[-(lb_market + 1):].mean(axis=1)
    if market_recent[0] <= 1e-8:
        return w
    market_ret = (market_recent[-1] - market_recent[0]) / market_recent[0]
    
    # 基于市场状态的暴露控制
    if market_ret < -0.08:
        exposure = 0.1  # 深度熊市，极度防守
    elif market_ret < -0.03:
        exposure = 0.4
    elif market_ret < 0:
        exposure = 0.7
    else:
        exposure = 1.0

    # 2. 因子计算 (短周期动量)
    lb_factor = min(10, n_days - 1)
    if lb_factor <= 1:
        return w
        
    recent = prices[-(lb_factor + 1):]
    p_start = recent[0, :]
    p_end = recent[-1, :]
    
    valid = (p_start > 1e-8) & (p_end > 1e-8)
    if not np.any(valid):
        return w
        
    returns = np.zeros(n_stocks)
    returns[valid] = (p_end[valid] - p_start[valid]) / p_start[valid]
    
    # 简单波动率惩罚
    ret_diff = np.diff(recent, axis=0)
    vol = np.std(ret_diff, axis=0)
    vol = np.where(vol < 1e-8, 1e-8, vol)
    
    # 风险调整得分
    score = returns / vol
    # 负动量给予强惩罚
    score[returns < 0] = -1.0
    
    # 3. 选股与目标权重
    # 只选正收益且排名前 10 的股票 (更少持仓，更集中，降低换手)
    positive_mask = returns > 0
    candidates = np.where(positive_mask)[0]
    
    if len(candidates) == 0:
        # 无正收益股票，保持空仓或极低暴露
        target_w = np.zeros(n_stocks)
    else:
        # 在正收益股票中按得分排序，取前 10 名
        sorted_candidates = candidates[np.argsort(score[candidates])[::-1][:10]]
        
        # 初始平均权重
        initial_w = 1.0 / len(sorted_candidates)
        
        # 构建目标权重
        target_w = np.zeros(n_stocks)
        target_w[sorted_candidates] = initial_w
        
        # 应用市场暴露
        target_w *= exposure
        
        # 单只上限 20%
        target_w = np.minimum(target_w, 0.20)
        
        # 归一化确保总权重不超过 exposure (实际上乘积后 sum 可能小于 exposure，这是允许的，即持有现金)
        # 如果 sum > exposure, 则缩放
        curr_sum = target_w.sum()
        if curr_sum > exposure:
            target_w *= (exposure / curr_sum)

    # 4. 降低换手率：滞回控制 (Hysteresis)
    # 核心逻辑：
    # 1. 如果新目标权重为 0，且旧权重 > 0，只有当旧权重较大时才卖出，或者强制卖出以释放资金给新赢家。
    #    为了降低换手，我们设定一个“持有底线”：如果新目标为0，但旧权重很小，保持不动？
    #    不，为了风险，新目标为0通常意味着动量破裂，应卖出。
    #    但是，频繁的进出导致换手。
    #    策略调整：
    #    - 买入缓冲：只有当 new_w > 0 且 (new_w > old_w + threshold_buy) 或 old_w==0 且 new_w > threshold_entry 时才买入/增持。
    #    - 卖出缓冲：只有当 new_w == 0 时卖出。如果 new_w > 0 但 new_w < old_w - threshold_sell，则调整到 new_w。
    #    - 保持：如果 new_w 和 old_w 接近，保持 old_w。
    
    threshold = 0.05  # 5% 的缓冲带
    
    final_w = np.zeros(n_stocks)
    
    for i in range(n_stocks):
        old_w = position[i]
        new_w = target_w[i]
        
        if old_w == 0:
            # 当前无持仓
            if new_w > 0:
                # 只有当目标权重显著大于0时才买入，避免极小仓位
                if new_w > 0.02: 
                    final_w[i] = new_w
                else:
                    final_w[i] = 0
            else:
                final_w[i] = 0
        else:
            # 当前有持仓
            if new_w == 0:
                # 目标为0，强制卖出 (动量失效)
                final_w[i] = 0
            elif new_w > old_w + threshold:
                # 显著增持
                final_w[i] = new_w
            elif new_w < old_w - threshold:
                # 显著减持
                final_w[i] = new_w
            else:
                # 权重变化不大，保持原仓位，避免微小调整
                final_w[i] = old_w

    # 5. 最终约束
    # 确保总权重不超过 exposure
    total_weight = final_w.sum()
    if total_weight > exposure:
        # 按比例缩减
        final_w *= (exposure / total_weight)
        
    # 再次检查单只上限
    final_w = np.minimum(final_w, 0.20)
    
    # 确保非负
    final_w = np.maximum(final_w, 0)
    
    return final_w