# scripts/contest_config.py
"""第二阶段评测的全局配置。想改参数只改这里。"""

# ---- 市场规模 ----
N_STOCKS = 5300
N_DAYS = 252
INITIAL_CASH = 10_000_000.0

# ---- 参赛模型参数覆盖（合并到 models_v4 的 extra 上）----
MODEL_OVERRIDES = {
    "Qwen3.8-27B":            {"max_tokens": 16384},
    "DeepSeek-V4-Flash-0731": {"max_tokens": 16384},
    "glm-5.3-flash":          {"max_tokens": 16384},
    "glm-5.3-flashx":         {"max_tokens": 16384},
    "Xing4.0-29B-A4B":        {"max_completion_tokens": 16384},
    "TeleChat3-438B-A24B":    {"max_completion_tokens": 16384},
    "GLM-4-9B":               {"max_tokens": 16384},
    "GLM-Z1-9B":              {"max_tokens": 16384},
    "DeepSeek-R1-8B":         {"max_tokens": 16384},
    "deepseek-v4.1-flash":    {"max_tokens": 65536, "reasoning_effort": "low"},
    "deepseek-v4-pro":        {"max_tokens": 65536},
    "claude-opus-5":          {"max_tokens": 16384},
    "grok-4.7":               {"max_tokens": 32768},
    "seed-2.1-turbo":         {"max_tokens": 16384},
}

# ---- 种子（从 contest_seeds.json 读）----
import json as _json
from pathlib import Path as _Path
_seed_file = _Path(__file__).parent.parent / "data" / "contest_seeds.json"
if _seed_file.exists():
    _sd = _json.loads(_seed_file.read_text(encoding="utf-8"))
    TRAIN_SEEDS = _sd["train_seeds"]
    TEST_SEED_GROUPS = _sd["test_seed_groups"]
else:
    raise FileNotFoundError(
        f"缺少 {_seed_file}。请先跑 scripts/pick_seeds_balanced.py")

# ---- 优化轮数 ----
N_ROUNDS = 4

# ---- 超时（秒）----
API_TIMEOUT = 180.0       # 单次 API 调用（reasoning 模型慢）
STRATEGY_TIMEOUT = 5.0
API_RETRY = 2
CODE_RETRY = 1

# ---- 输出路径 ----
from pathlib import Path
ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "data" / "contest_results"
STRATEGIES_DIR = ROOT / "data" / "strategies"
LOGS_DIR = ROOT / "data" / "contest_logs"
for d in (RESULTS_DIR, STRATEGIES_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- 评分 ----
TEST_WEIGHT = 0.90
TRAIN_WEIGHT = 0.10

METRIC_WEIGHTS = {
    "avg_return":   50,
    "sharpe":       20,
    "max_drawdown": 15,
    "turnover":     10,
    "variance":      5,
}

OVERFIT_THRESHOLD = 0.5
OVERFIT_PENALTY = 0.30