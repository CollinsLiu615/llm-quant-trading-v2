def strategy(prices, news, position, cash):
    import numpy as np
    n_days, n_stocks = prices.shape
    w = np.zeros(n_stocks, dtype=float)
    try:
        txt = " ".join(news) if news is not None else ""
        s = sum(txt.count(k) for k in ["利好","增长","上升","突破","回暖","超预期"]) - sum(txt.count(k) for k in ["利空","下跌","暴跌","风险","衰退","暴雷","危机","大跌"])
        last = prices[-1]
        valid = np.isfinite(last) & (last>0) & (last<1e6)
        valid &= np.sum(np.isfinite(prices), axis=0) >= 40
        valid &= np.sum(np.isfinite(prices[-5:]), axis=0) >= 4
        rets = prices[1:]/prices[:-1]-1
        rets[~np.isfinite(rets)] = np.nan
        rets[(rets>0.35)|(rets<-0.35)] = np.nan
        vol = np.nanstd(rets[-20:], axis=0)
        p20 = prices[-21] if n_days>=21 else prices[0]
        mom = np.where(np.isfinite(p20)&(p20>0)&np.isfinite(last), last/p20-1, np.nan)
        ma = np.nanmean(prices[-20:], axis=0)
        dist = np.where(np.isfinite(ma)&(ma>0), last/ma-1, np.nan)
        mx = np.nanmax(prices[-20:], axis=0)
        dd = np.where(np.isfinite(mx)&(mx>0), last/mx-1, 0)
        score = mom/(vol+0.02) - 1.5*np.maximum(dist-0.12,0) - np.maximum(-dist-0.06,0)
        score[~np.isfinite(score)] = -1e9
        score[~valid] = -1e9
        score[(vol<0.005)|(vol>0.05)|(mom<0.01)|(mom>0.5)|(dd<-0.15)] = -1e9
        mm = np.nanmean(mom[valid]) if np.sum(valid)>50 else 0
        tot = 0.9
        if s<=-2 or mm<-0.03: tot=0.1
        elif s==-1 or mm<-0.015: tot=0.3
        elif mm<0: tot=0.5
        if tot<0.15: return w
        held = position>1e-6
        score[held & (score>-1e8)] += 0.12
        order = np.argsort(-score)
        sel = []
        for i in order:
            if held[i] and score[i]>-1e8:
                sel.append(i)
                if len(sel)>=10: break
        for i in order:
            if i not in sel and score[i]>-1e8:
                sel.append(i)
            if len(sel)>=10: break
        if not sel: return w
        sc = score[sel]
        sc = np.maximum(sc,0)+0.01
        if s<0: sc = np.ones_like(sc)
        wsel = sc/sc.sum()*tot
        wsel = np.minimum(wsel,0.2)
        wsel = wsel/wsel.sum()*tot
        for k, idx in enumerate(sel): w[idx]=wsel[k]
        try:
            if np.sum(position)>0:
                pw = position/np.sum(position)*np.sum(w) if np.sum(position)>1.5 else position
                if np.sum(np.abs(w-pw))>0.5: w=0.6*w+0.4*pw
        except: pass
        return w
    except: return w