import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    
    # 计算过去5天的收益率
    ret = prices[-1, :] / prices[-6, :] - 1.0
    
    # 筛选出过去5天有正收益的股票
    pos = ret > 0
    if not np.any(pos):
        return w
    
    # 新闻情绪评分
    score = np.zeros(n_stocks)
    for i in range(n_days):
        for j in range(n_stocks):
            if pos[j]:
                text = news[i][j] if j < len(news[i]) else ""
                if "利好" in text or "上涨" in text or "突破" in text:
                    score[j] += 1
                elif "利空" in text or "下跌" in text or "破位" in text:
                    score[j] -= 1
    
    # 取得分最高的20只股票
    top_idx = np.argsort(score)[::-1][:20]
    
    # 分配权重：先给最高分的股票分配最大权重，依次递减
    max_weight = 0.2
    remaining = 1.0
    for idx in top_idx:
        if remaining >= max_weight:
            w[idx] = max_weight
            remaining -= max_weight
        else:
            w[idx] = remaining
            break
    
    # 确保总权重不超过1
    total = np.sum(w)
    if total > 1.0:
        w = w / total
    
    return w