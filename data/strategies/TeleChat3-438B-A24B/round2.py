import numpy as np

def strategy(prices, news, position, cash):
    n = prices.shape[1]
    last = prices[-1]
    ret = np.diff(np.log(prices), axis=0)
    
    # Market regime: bull/bear based on recent market return vs vol
    look = min(20, ret.shape[0])
    mkt_ret = ret[-look:].mean()
    mkt_vol = ret[-look:].std() + 1e-9
    regime = np.tanh(mkt_ret / mkt_vol)
    bear = (1 - regime) / 2
    pos_scale = 1.0 - 0.6 * bear
    
    # Features per stock
    mom = ret[-5:].mean(axis=0) if ret.shape[0] >= 5 else ret.mean(axis=0)
    vol = ret.std(axis=0) + 1e-9
    risk_adj = mom / vol
    
    # News sentiment
    pw = ['涨','利好','突破','增长','收购','业绩','上涨','涨停','大涨','超预期','回购','增持','分红','盈利','走强']
    nw = ['跌','利空','下跌','亏损','暴雷','违规','退市','减持','风险','暴跌','跌停','预警','爆仓','问询','立案']
    txt = ''.join(news)
    sent = sum(w in txt for w in pw) - sum(w in txt for w in nw)
    sent_b = 1.0 + 0.2 * np.tanh(sent)
    
    score = risk_adj * sent_b * pos_scale
    valid = last > 0
    score = np.where(valid, score, -np.inf)
    
    # Top picks
    idx = np.argsort(-score)[:20]
    raw = score[idx]
    s = raw.sum()
    w = np.zeros(20)
    if s > 0:
        wr = raw / s
        ex = np.maximum(wr - 0.15, 0)
        wr = np.minimum(wr, 0.15)
        if ex.sum() > 0:
            wr[ex > 0] += ex[ex > 0] / np.count_nonzero(ex)
        w = wr
    full = np.zeros(n)
    full[idx] = w
    
    # Stop-loss: cut if recent cumulative return < -10%
    if ret.shape[0] >= 5:
        cum = np.minimum.accumulate(ret[-5:])
        sl_mask = valid & (cum.min(axis=0) < -0.10)
        full[sl_mask] *= 0.2
    
    # Budget
    cur_val = (position * last).sum() + cash
    if cur_val <= 0:
        return full
    need = np.maximum(full * cur_val - position * last, 0).sum()
    if need > cash:
        full *= cash / (need + 1e-9)
    t = full.sum()
    if t > 1:
        full /= t
    return full