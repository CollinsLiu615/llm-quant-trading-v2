import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    
    # 计算最近5天的收益率：从第5天到第1天
    if n_days < 6:
        return w
    
    # 使用倒数第5天作为基准日，计算从第5天到第1天的收益率
    # 注意：prices[-5] 是倒数第5天，prices[-1] 是最后一天
    ret = prices[-1, :] / prices[-5, :] - 1.0
    
    # 筛选出最近5天有正收益的股票
    pos = ret > 0
    if not np.any(pos):
        return w
    
    # 初始化新闻情绪评分
    score = np.zeros(n_stocks)
    positive_keywords = ["利好", "上涨", "突破", "增长", "盈利", "超预期", "大涨"]
    negative_keywords = ["利空", "下跌", "破位", "亏损", "下滑", "暴跌", "风险"]
    
    # 处理新闻数据：假设 news 是二维列表，news[i][j] 是第 i 天第 j 只股票的新闻文本
    # 为了兼容，我们检查 news 的类型
    if isinstance(news, list) and len(news) > 0 and isinstance(news[0], list):
        # 二维列表
        for i in range(n_days):
            for j in range(n_stocks):
                if pos[j]:
                    text = news[i][j] if j < len(news[i]) else ""
                    if any(keyword in text for keyword in positive_keywords):
                        score[j] += 1
                    elif any(keyword in text for keyword in negative_keywords):
                        score[j] -= 1
    else:
        # 如果 news 是字符串列表，可能每个字符串包含多只股票的信息？
        # 这里我们暂时跳过新闻处理，因为无法确定
        pass
    
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