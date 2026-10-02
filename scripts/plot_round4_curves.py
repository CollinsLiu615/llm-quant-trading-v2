# scripts/plot_round4_curves.py
"""Round 4 权益曲线（改进配色/线粗/分辨率）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "PingFang SC"]
rcParams["axes.unicode_minus"] = False

from code_agent import CodeAgent
from stock_sim_v4 import MarketConfig, MarketSimulator, EnvConfig, run_backtest
import contest_config as cfg
from models_v4 import MODELS_ORDERED


SEED = 2513

# 为每个模型分配对比强烈的颜色 + 不同线型
STYLE = {
    # 6 个真实策略 —— 各占一个差异明显的色系
    "claude-opus-5":            {"color": "#E63946", "lw": 3.0, "ls": "-"},   # 红
    "seed-2.1-turbo":           {"color": "#F77F00", "lw": 3.0, "ls": "-"},   # 亮橙
    "deepseek-v4-pro":          {"color": "#06A77D", "lw": 3.0, "ls": "-"},   # 绿
    "DeepSeek-V4-Flash-0731":   {"color": "#7209B7", "lw": 2.6, "ls": "-"},   # 深紫
    "Qwen3.8-27B":              {"color": "#1D3557", "lw": 2.6, "ls": "-"},   # 深蓝
    "DeepSeek-R1-8B":           {"color": "#FFB703", "lw": 2.6, "ls": "-"},   # 金
    # 空仓模型 —— 虚线
    "glm-5.3-flashx":           {"color": "#B0B0B0", "lw": 1.8, "ls": ":"},
    "Xing4.0-29B-A4B":          {"color": "#A8DADC", "lw": 1.8, "ls": ":"},
    "TeleChat3-438B-A24B":      {"color": "#F4A6C0", "lw": 1.8, "ls": ":"},
    "GLM-4-9B":                 {"color": "#B8D4A0", "lw": 1.8, "ls": ":"},
    "deepseek-v4.1-flash":      {"color": "#FDDCB5", "lw": 1.8, "ls": ":"},
}


def main():
    print(f"生成市场 seed={SEED}...")
    mkt = MarketSimulator(
        MarketConfig(n_stocks=cfg.N_STOCKS, n_days=cfg.N_DAYS, seed=SEED)).generate()

    fig, ax = plt.subplots(figsize=(16, 9), dpi=150)

    # 市场均值——黑粗虚线
    bench = mkt.prices.mean(axis=1)
    ax.plot(bench / bench[0] * 100, label="市场均值", color="black",
            linestyle=(0, (4, 4)), linewidth=3, zorder=10, alpha=0.85)

    for model in MODELS_ORDERED:
        code_file = cfg.STRATEGIES_DIR / model / "round4.py"
        if not code_file.exists():
            continue
        code = code_file.read_text(encoding="utf-8")
        try:
            agent = CodeAgent(code, n_stocks=cfg.N_STOCKS,
                              timeout=cfg.STRATEGY_TIMEOUT)
        except Exception as e:
            print(f"  {model}: 编译失败 {str(e)[:50]}")
            continue
        res = run_backtest(agent, mkt, EnvConfig(initial_cash=cfg.INITIAL_CASH))
        eq = res.equity_curve
        st = STYLE.get(model, {"color": "gray", "lw": 1.5, "ls": "-"})
        ax.plot(eq / eq[0] * 100, label=model,
                color=st["color"], linewidth=st["lw"], linestyle=st["ls"],
                zorder=5, alpha=0.95)
        print(f"  {model}: {res.metrics['total_return']*100:+.2f}%")

    ax.set_xlabel("交易日", fontsize=14)
    ax.set_ylabel("净值（初始=100）", fontsize=14)
    ax.set_title(f"Round 4 权益曲线 — 测试种子 {SEED}，5300股 × 252天",
                 fontsize=16, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11, framealpha=0.9, ncol=2)
    ax.grid(alpha=0.3, linestyle=":")
    ax.tick_params(labelsize=12)
    out = ROOT / "figures" / f"round4_curves_seed{SEED}.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\n✅ 图保存: {out}")


if __name__ == "__main__":
    main()
