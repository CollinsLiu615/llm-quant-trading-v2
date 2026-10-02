import numpy as np

def strategy(prices, news, position, cash):
    n = prices.shape[1]
    last = prices[-1]
    # returns
    ret = np.diff(np.log(prices), axis=0)  # 59xN
    # features
    mom_short = ret[-5:].mean(axis=0) if ret.shape[0]>=5 else ret.mean(axis=0)
    mom_long = ret.mean(axis=0)
    vol = ret.std(axis=0)
    # avoid division by zero
    risk_adj = mom_short / (vol + 1e-9)
    # news sentiment
    pos_words = ['涨','利好','突破','增长','收购','业绩','上涨','涨停','大涨','超预期','回购','增持','分红','盈利','走强']
    neg_words = ['跌','利空','下跌','亏损','暴雷','违规','退市','减持','风险','暴跌','跌停','预警','爆仓','问询','立案']
    text = ''.join(news)
    pos = sum(w in text for w in pos_words)
    neg = sum(w in text for w in neg_words)
    sent = pos - neg
    sent_boost = 1.0 + 0.2*np.tanh(sent)
    score = risk_adj * sent_boost
    # mask invalid
    valid = last > 0
    score = np.where(valid, score, -np.inf)
    # rank
    idx = np.argsort(-score)[:20]
    top_score = score[idx]
    # normalize weights with cap 0.2
    raw = top_score
    w = np.zeros(n)
    # simple capping: if any raw >0.2, set to 0.2 and add excess to next
    # but easier: normalize then cap and renormalize
    s = raw.sum()
    if s>0:
        w_raw = raw/s
    else:
        w_raw = np.zeros(20)
    # cap at 0.2
    excess = np.maximum(w_raw - 0.2, 0)
    w_raw = np.minimum(w_raw, 0.2)
    # redistribute excess to others in top list (excluding capped)
    if excess.sum()>0:
        others = w_raw[excess>0]
        add = excess[excess>0]/len(others)
        w_raw[excess>0] += add
    w[idx] = w_raw
    # budget constraint: ensure feasible with cash and positions
    cur_val = (position*last).sum() + cash
    if cur_val<=0:
        return w
    # target total value if sum=1 is cur_val; but if we want sum=1, need cash for net buys.
    # compute required additional cash for target vs current position
    target_val = cur_val  # sum weights=1
    # current position value in selected
    cur_pos_sel = (position[idx]*last[idx]).sum()
    need_buy = np.maximum(w[idx]*target_val - position[idx]*last[idx], 0).sum()
    if need_buy > cash:
        # scale down buy side only? simpler scale all weights
        scale = (cash)/(need_buy + 1e-9)
        w[idx] *= scale
    # ensure sum<=1
    total = w.sum()
    if total>1:
        w = w/total
    return w