# scripts/pick_seeds_balanced.py
"""从大池子里挑出覆盖牛熊的种子（分层抽样）。不调 API，纯本地。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from stock_sim_v4 import MarketConfig, MarketSimulator

N_STOCKS = 5300
N_DAYS = 252
POOL = list(range(100, 1500))   # 候选池 100~1499
N_TRAIN = 6
N_TEST_GROUPS = 3
N_TEST_PER_GROUP = 6


def market_median(seed):
    mkt = MarketSimulator(MarketConfig(n_stocks=N_STOCKS, n_days=N_DAYS, seed=seed)).generate()
    return float(np.median(mkt.prices[-1] / mkt.prices[0] - 1))


def main():
    print(f"扫描 {len(POOL)} 个候选 seed（本地计算，不调 API）...")
    results = []
    for i, s in enumerate(POOL):
        results.append((s, market_median(s)))
        if (i + 1) % 200 == 0:
            print(f"  已扫 {i+1}/{len(POOL)}")
    results.sort(key=lambda x: x[1])
    print(f"扫描完成\n")

    meds = np.array([r for _, r in results])
    print(f"池子统计：中位数 {np.median(meds)*100:+.1f}%，范围 {meds.min()*100:+.1f}% ~ {meds.max()*100:+.1f}%")
    print()

    # 训练：覆盖 10%, 30%, 50%, 70%, 90%, 95% 分位
    quantiles = [0.10, 0.30, 0.50, 0.70, 0.90, 0.95]
    n = len(results)
    train = []
    for q in quantiles:
        idx = int(n * q)
        train.append(results[min(idx, n-1)][0])

    # 训练种子去重 + 排序
    train = sorted(set(train))
    # 如果去重后不足 6 个，补几个中间分位
    if len(train) < N_TRAIN:
        for q in [0.15, 0.40, 0.60, 0.80, 0.85, 0.92]:
            idx = int(n * q)
            s = results[min(idx, n-1)][0]
            if s not in train:
                train.append(s)
            if len(train) >= N_TRAIN:
                break
    train = train[:N_TRAIN]

    print(f"训练种子（覆盖牛熊，各分位）：")
    for s in train:
        m = market_median(s)
        print(f"  {s:>5}: {m*100:+6.1f}%")
    print()

    # 测试：排除训练，随机抽 3 组
    rng = np.random.default_rng(20261002)  # 固定种子，可复现
    available = [s for s, _ in results if s not in train]
    rng.shuffle(available)

    test_groups = []
    used = set(train)
    idx = 0
    for _ in range(N_TEST_GROUPS):
        grp = []
        while len(grp) < N_TEST_PER_GROUP and idx < len(available):
            s = available[idx]
            idx += 1
            if s in used:
                continue
            used.add(s)
            grp.append(s)
        test_groups.append(sorted(grp))

    print(f"测试种子（3 组 × 6，随机）：")
    for i, g in enumerate(test_groups, 1):
        print(f"  组{i}:")
        for s in g:
            m = market_median(s)
            print(f"    {s:>5}: {m*100:+6.1f}%")
        print()

    # 落盘
    out = ROOT / "data" / "contest_seeds.json"
    data = {
        "train_seeds": train,
        "test_seed_groups": test_groups,
        "note": "分层抽样 + 随机分组，覆盖牛熊全光谱",
    }
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 写入 {out}")


if __name__ == "__main__":
    main()