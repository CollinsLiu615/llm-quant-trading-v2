import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    
    if n_days < 3:
        return w
        
    # 1. 市场状态识别与防守
    lookback_market = min(60, n_days - 1)
    if lookback_market <= 1:
        return w
        
    recent_market = prices[-(lookback_market + 1):]
    market_return = (recent_market[-1, :] - recent_market[0, :]) / (recent_market[0, :] + 1e-8)
    avg_market_return = np.mean(market_return)
    
    # 熊市降仓：大幅下跌时极度保守，温和下跌时适度保守
    if avg_market_return < -0.10:
        exposure = 0.1
    elif avg_market_return < -0.03:
        exposure = 0.4
    elif avg_market_return < 0:
        exposure = 0.7
    else:
        exposure = 1.0
        
    # 2. 因子计算
    lookback = min(20, n_days - 1)
    if lookback <= 0:
        return w
        
    recent = prices[-(lookback + 1):]
    prices_last = recent[-1, :]
    prices_start = recent[0, :]
    
    valid = (prices_start > 0) & (prices_last > 0)
    
    # 动量
    momentum = np.zeros(n_stocks)
    momentum[valid] = (prices_last[valid] - prices_start[valid]) / prices_start[valid]
    
    # 波动率
    returns = np.diff(recent, axis=0)
    volatility = np.std(returns, axis=0)
    volatility = np.where(volatility < 1e-8, 1e-8, volatility)
    
    # 风险调整后的收益 (Sharpe-like)
    sharpe_like = momentum / volatility
    # 惩罚负动量
    sharpe_like[momentum <= 0] = -1
    
    # 3. 选股与权重分配
    # 选择前 20 只
    top_indices = np.argsort(sharpe_like)[::-1][:20]
    
    # 分配权重，确保单只不超过 20%
    # 初始平均分配，然后归一化
    w_new = np.zeros(n_stocks)
    if len(top_indices) > 0:
        w_new[top_indices] = 1.0 / len(top_indices)
    
    # 限制单只最大权重为 15% (留出缓冲，避免触及 20% 上限导致频繁调整)
    w_new = np.minimum(w_new, 0.15)
    
    # 归一化
    total_w = w_new.sum()
    if total_w > 0:
        w_new = w_new / total_w
    else:
        w_new = np.zeros(n_stocks)
        
    # 应用市场暴露
    w_new = w_new * exposure
    
    # 4. 降低换手率：引入持仓缓冲
    # 如果当前持仓权重 > 0，且新目标权重也为 0，或者排名大幅下滑，才卖出
    # 如果当前持仓权重为 0，但新目标权重 > 0，才买入
    # 这里使用一个简单的阈值逻辑：
    # 只有当目标权重与当前权重的差异超过一定阈值（如 2%），才进行调整
    # 否则保持原有权重（或保持为0）
    
    threshold = 0.02
    
    w_final = np.zeros(n_stocks)
    
    for i in range(n_stocks):
        old_w = position[i]
        new_w = w_new[i]
        
        if old_w > 0:
            # 如果新权重很低，或者旧权重很高但新权重显著降低，则调整
            # 为了减少换手，只有当新权重 < 旧权重 - threshold 时才卖出部分或全部
            # 简化：如果新权重 < threshold，则清仓；否则保持旧权重直到显著变化
            # 更激进的减少换手策略：如果新权重 >= old_w * 0.8，保持 old_w (不增加也不大幅减少)
            # 如果新权重 <= 0，清仓
            # 如果新权重 > old_w，增加到 new_w (买入通常更灵活，因为我们要捕捉新机会)
            # 这里采取：卖出滞后。只有当 new_w < old_w - threshold 时才下调。
            # 如果 new_w >= old_w - threshold，则保持 old_w (防止微小的权重波动导致卖出)
            # 但如果是牛市且新股票进入池子，我们需要买入。
            
            if new_w == 0:
                # 如果新目标为0，且旧权重不为0，为了减少换手，可以设一个保留底仓或直接清仓
                # 策略诊断指出换手高，直接清仓可能导致高频交易。
                # 但如果不卖，坏股票会占用仓位。
                # 折中：如果 new_w == 0，强制卖出 (因为不在前20名，通常表现差)
                w_final[i] = 0
            elif new_w < old_w - threshold:
                # 权重下降超过阈值，调整到目标
                w_final[i] = new_w
            else:
                # 权重变化不大，保持旧权重，避免微小调整带来的交易成本
                # 注意：这可能导致总权重超过 exposure 限制，需要后续归一化
                w_final[i] = old_w
        else:
            # 当前无持仓
            if new_w > threshold:
                # 新股票，买入
                w_final[i] = new_w
            else:
                # 权重太小，忽略
                w_final[i] = 0
                
    # 5. 最终约束与归一化
    # 确保总暴露不超过 exposure
    total_final = w_final.sum()
    if total_final > exposure + 0.01:
        # 按比例缩减，以符合市场暴露限制
        w_final = w_final * (exposure / total_final)
        
    # 限制单只权重上限 20%
    w_final = np.minimum(w_final, 0.20)
    
    # 如果缩减后单只权重过高，可能需要再次归一化，但通常 exposure <= 1，且初始分布均匀，问题不大
    # 确保非负
    w_final = np.maximum(w_final, 0)
    
    # 最后归一化，确保 sum <= 1
    final_sum = w_final.sum()
    if final_sum > 1:
        w_final = w_final / final_sum
        
    return w_final