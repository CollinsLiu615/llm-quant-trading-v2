import numpy as np

def strategy(prices, news, position, cash):
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks)
    if n_days < 21 or n_stocks < 20:
        return w

    prices = np.where(np.isfinite(prices), prices, 0.0)
    last = prices[-1]
    prev = prices[-2]
    ref20 = prices[-21]
    ref5 = prices[-6]

    valid = (last > 0) & (prev > 0) & (ref20 > 0) & (ref5 > 0)
    valid &= (last >= 2.0) & (last <= 1000.0)
    valid &= np.all(prices[-20:] > 0, axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        ret1 = last / prev - 1.0
        ret_matrix = prices[1:] / prices[:-1] - 1.0
    ret_matrix = np.where(np.isfinite(ret_matrix), ret_matrix, 0.0)

    valid &= (ret1 > -0.095) & (ret1 < 0.095)

    vol = np.std(ret_matrix[-20:], axis=0)
    valid &= (vol > 0) & np.isfinite(vol)

    if not np.any(valid):
        return w

    total_asset = cash + np.sum(position * last)
    if total_asset <= 0:
        total_asset = 1e6

    pos_words = ['利好','增长','突破','盈利','回购','降准','降息','支持','签约','中标','上涨','创新高','改革','政策','放宽','复苏','超预期','扭亏','分红','增持']
    neg_words = ['利空','下滑','亏损','处罚','调查','退市','风险','减持','暴跌','违约','下调','警告','衰退','收紧','监管','跌停','警示','诉讼','债务','危机']
    news_text = ' '.join([str(x).lower() for x in news])
    pos_count = sum(news_text.count(word) for word in pos_words)
    neg_count = sum(news_text.count(word) for word in neg_words)
    sentiment = pos_count - neg_count

    target_exposure = 0.85 + 0.03 * max(-3, min(3, sentiment))
    target_exposure = min(1.0, max(0.4, target_exposure))

    mom20 = last / ref20 - 1.0
    mom5 = last / ref5 - 1.0

    def zscore(arr, mask):
        vals = arr[mask]
        mu = np.mean(vals)
        sigma = np.std(vals)
        if sigma < 1e-8:
            return np.zeros_like(arr)
        out = np.zeros_like(arr)
        out[mask] = (arr[mask] - mu) / sigma
        return out

    z20 = zscore(mom20, valid)
    z5 = zscore(mom5, valid)
    zv = zscore(vol, valid)
    held = (position > 0).astype(float)

    score = 0.5 * z20 - 0.2 * z5 - 0.2 * zv + 0.25 * held

    min_weight = target_exposure / 20.0
    max_price = total_asset * min_weight / 100.0
    valid &= (last <= max_price)

    if not np.any(valid):
        return w

    valid_idx = np.where(valid)[0]
    score_valid = score[valid]
    K = min(20, len(valid_idx))
    if K == 0:
        return w

    top_local = np.argsort(-score_valid)[:K]
    selected = valid_idx[top_local]

    weight = target_exposure / K
    if weight > 0.20:
        weight = 0.20

    w[selected] = weight
    return w