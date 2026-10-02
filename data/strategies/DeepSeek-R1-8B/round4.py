import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    # Calculate recent price volatility (last 5 days)
    if n_days < 5:
        window = n_days
    else:
        window = 5
    
    # Calculate log returns for volatility
    log_returns = np.zeros((n_stocks,))
    if n_days >= window:
        last_prices = prices[-window:, :]
        for i in range(n_stocks):
            if window > 1:
                log_return = np.log(last_prices[1:, i] / last_prices[:-1, i])
                log_returns[i] = np.mean(log_return) if len(log_return) > 0 else 0
            else:
                log_returns[i] = 0
    
    volatilities = np.std(log_returns) if n_days >= window else 0
    
    # Calculate momentum (last 5 days price change)
    momentum = np.zeros(n_stocks)
    if n_days >= 2:
        momentum = prices[-1, :] - prices[-2, :]
    
    # Calculate overall market sentiment from news
    sentiment_score = 0
    positive_words = ['利好', '上涨', '增长', '突破', '创新高', '业绩增长', '盈利']
    negative_words = ['利空', '下跌', '下跌', '亏损', '破发', '风险', '危机']
    
    for title in news:
        count_positive = sum(title.count(word) for word in positive_words)
        count_negative = sum(title.count(word) for word in negative_words)
        sentiment_score += (count_positive - count_negative)
    
    # Market state identification (bullish/bearish)
    market_bullish = False
    if n_days >= 10:
        # Check if market is bullish (last 10 days prices are higher than first 10 days)
        if prices[-1, 0] > prices[-11, 0] and np.mean(prices[-10:, 0]) > np.mean(prices[-20:-10, 0]):
            market_bullish = True
    
    # Determine bear market condition
    bear_market = not market_bullish
    
    # Combine factors into a score for each stock
    scores = np.zeros(n_stocks)
    for i in range(n_stocks):
        # Normalize volatility (lower is better)
        vol_normalized = 1 / (1 + volatilities * 100) if volatilities > 0 else 1
        # Momentum factor (higher is better)
        mom_factor = momentum[i] / (prices[-1, i] if n_days >= 2 else 1)
        # Sentiment factor (higher is better)
        sent_factor = 1 + sentiment_score * 0.01
        
        # Adjust for market state - bear market reduction
        if bear_market:
            vol_normalized *= 0.5
            mom_factor *= 0.7
            sent_factor *= 0.8
        
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
        # Maximum weight per stock is 20%, reduced in bear market
        max_weight = 0.15 if bear_market else 0.2
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
    
    # Apply position sizing with stop-loss
    final_weights = np.zeros(n_stocks)
    for i in range(n_stocks):
        if position[i] > 0.001:  # Only consider stocks with significant position
            # Calculate stop-loss price (3% below current price for bear market, 5% otherwise)
            stop_loss_multiplier = 0.97 if bear_market else 0.95
            stop_loss_price = prices[-1, i] * stop_loss_multiplier
            # If current price is below stop-loss, reduce position
            if prices[-1, i] < stop_loss_price:
                final_weights[i] = max(0, position[i] * 0.3)  # Reduce position by 70% in bear market
            else:
                final_weights[i] = target_weights[i]
        else:
            final_weights[i] = target_weights[i]
    
    return final_weights