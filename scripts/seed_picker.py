# scripts/seed_picker.py
"""随机抽取训练/测试种子，写入 data/contest_seeds.json。
只跑一次，之后 run_contest.py 读取这份文件，保证可复现。
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "contest_seeds.json"

N_TRAIN = 6
N_TEST_GROUPS = 3
N_TEST_PER_GROUP = 6

# 排除太小的数字（避免和早期实验撞），并排除 1~50（已用于诊断）
EXCLUDE = set(range(1, 51))


def pick(seed_pool_size=10000, rng_seed=None):
    rng = random.Random(rng_seed)

    def sample(n):
        while True:
            s = rng.randint(51, seed_pool_size)
            if s not in EXCLUDE:
                return s

    train = []
    used = set(EXCLUDE)
    while len(train) < N_TRAIN:
        s = sample(N_TRAIN)
        if s in used:
            continue
        used.add(s)
        train.append(s)

    test_groups = []
    for _ in range(N_TEST_GROUPS):
        grp = []
        while len(grp) < N_TEST_PER_GROUP:
            s = sample(1)
            if s in used:
                continue
            used.add(s)
            grp.append(s)
        test_groups.append(sorted(grp))

    return sorted(train), test_groups


def main():
    # 可通过命令行传种子：python3 seed_picker.py 12345
    rng_seed = int(sys.argv[1]) if len(sys.argv) > 1 else None
    train, test_groups = pick(rng_seed=rng_seed)

    data = {
        "train_seeds": train,
        "test_seed_groups": test_groups,
        "rng_seed": rng_seed,
        "note": "随机抽取，落盘后不改，保证评测可复现",
    }
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"✅ 写入 {OUT}")
    print(f"训练种子: {train}")
    for i, g in enumerate(test_groups, 1):
        print(f"测试组 {i}: {g}")


if __name__ == "__main__":
    main()