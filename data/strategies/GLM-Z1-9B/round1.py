import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    if n_days < 5:
        return np.zeros(n_stocks)
    prev = prices[-5]
    curr = prices[-1]
    returns = (curr - prev) / prev
    top = returns.argsort()[-20:][::-1]
    w = np.zeros(n_stocks)
    w[top] = 0.05
    return w