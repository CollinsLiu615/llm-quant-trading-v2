import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    # Calculate recent price volatility (last 5 days)
    if n_days < 5:
        # If not enough days, use all available days
        window = n_days
    else:
        window = 5
    
    # Calculate log returns for volatility
    log_returns = np.log(prices[-window:] / np.roll(prices[-window:], 1, axis=0))
    # Calculate standard deviation of log returns (volatility)
    volatilities = np.std(log_returns, axis=0)
    
    # Calculate momentum (last 5 days price change)
    momentum = prices[-1] - prices[-2] if n_days >= 2 else 0
    
    # Calculate overall market sentiment from news
    sentiment_score = 0
    positive_words = ['利好', '上涨', '增长', '突破', '创新高', '业绩增长', '盈利']
    negative_words = ['利空', '下跌', '下跌', '亏损', '破发', '风险', '危机']
    
    for title in news:
        count_positive = sum(title.count(word) for word in positive_words)
        count_negative = sum(title.count(word) for word in negative_words)
        sentiment_score += (count_positive - count_negative)
    
    # Combine factors into a score for each stock
    # Higher score means better investment opportunity
    scores = np.zeros(n_stocks)
    for i in range(n_stocks):
        # Normalize volatility (lower is better)
        vol_normalized = 1 / (1 + volatilities[i] * 100) if volatilities[i] > 0 else 1
        # Momentum factor (higher is better)
        mom_factor = momentum / (prices[-1] if n_days >= 2 else 1) if n_days >= 2 else 1
        # Sentiment factor (higher is better)
        sent_factor = 1 + sentiment_score * 0.01
        
        # Combine factors with weights
        scores[i] = vol_normalized * 0.4 + mom_factor * 0.3 + sent_factor * 0.3
    
    # Sort stocks by score and select top 20
    sorted_indices = np.argsort(scores)[::-1]
    top_stocks = sorted_indices[:20]
    
    # Calculate current portfolio value
    current_value = 0
    for i in range(n_stocks):
        current_value += position[i] * prices[-1, i]
    
    # Calculate target weights
    target_weights = np.zeros(n_stocks)
    total_weight = 0
    
    for i in top_stocks:
        # Maximum weight per stock is 20%
        max_weight = 0.2
        # Calculate weight based on score
        weight = min(max_weight, scores[i] * 0.5)
        # Ensure weight is at least 0.01% to avoid too small positions
        weight = max(weight, 0.0001)
        
        # Add weight to target
        target_weights[i] = weight
        total_weight += weight
    
    # Normalize weights to sum to 1
    if total_weight > 0:
        target_weights = target_weights / total_weight
    
    # Ensure weights are within [0, 1]
    target_weights = np.clip(target_weights, 0, 1)
    
    return target_weights