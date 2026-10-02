# scripts/dry_run.py
"""小规模试跑：2 个免费模型 + 1 轮 + 2 训练种子 + 1 组 2 测试种子。
验证流程是否跑通，不花钱。"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_contest

# 覆盖全局配置
run_contest.MODELS_ORDERED = ["Qwen3.8-27B", "DeepSeek-V4-Flash-0731"]
run_contest.cfg.N_ROUNDS = 1
run_contest.cfg.TRAIN_SEEDS = [407, 527]      # 一熊一牛
run_contest.cfg.TEST_SEED_GROUPS = [[120, 761]]

run_contest.main()