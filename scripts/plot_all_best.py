# scripts/plot_all_best.py
"""所有模型最佳轮次的权益曲线对比（含 Xing）。"""
import sys, json
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from pathlib import Path
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

# 颜色：每个模型独占一色
STYLE = {
    "claude-opus-5":            {"color": "#E63946", "lw": 2.8, "ls": "-"},
    "seed-2.1-turbo":           {"color": "#F77F00", "lw": 2.8, "ls": "-"},
    "deepseek-v4-pro":          {"color": "#06A77D", "lw": 2.8, "ls": "-"},
    "glm-5.3-flashx":           {"color": "#7209B7", "lw": 2.4, "ls": "-"},
    "grok-4.7":                 {"color": "#FFB703", "lw": 2.4, "ls": "-"},
    "gpt-5.6-sol":              {"color": "#1D3557", "lw": 2.4, "ls": "-"},
    "deepseek-v4.1-flash":      {"color": "#9D0208", "lw": 2.4, "ls": "-"},
    "DeepSeek-V4-Flash-0731":   {"color": "#457B9D", "lw": 2.4, "ls": "-"},
    "Qwen3.8-27B":              {"color": "#2A9D8F", "lw": 2.4, "ls": "-"},
    "DeepSeek-R1-8B":           {"color": "#8B5A2B", "lw": 2.4, "ls": "-"},
    "TeleChat3-438B-A24B":      {"color": "#C77DFF", "lw": 2.4, "ls": "-"},
    "Xing4.0-29B-A4B":          {"color": "#F72585", "lw": 2.8, "ls": "-"},
}


def main():
    # 读最佳轮次
    best = {}
    f = cfg.RESULTS_DIR / "final_scores.json"
    if f.exists():
        for r in json.loads(f.read_text(encoding="utf-8")):
            best[r["model"]] = r["best_round"]
    best["Xing4.0-29B-A4B"] = 3  # 手动补

    print(f"生成市场 seed={SEED}...")
    mkt = MarketSimulator(MarketConfig(n_stocks=cfg.N_STOCKS,
                                       n_days=cfg.N_DAYS, seed=SEED)).generate()

    fig, ax = plt.subplots(figsize=(18, 10), dpi=150)

    # 市场基准——黑粗虚线
    bench = mkt.prices.mean(axis=1)
    ax.plot(bench / bench[0] * 100, label="市场均值",
            color="black", linestyle=(0, (4, 4)),
            linewidth=3.2, zorder=10, alpha=0.85)

    # 按最终收益排序，图例一致
    all_models = MODELS_ORDERED + ["Xing4.0-29B-A4B"]
    results = []

    for model in all_models:
        rnd = best.get(model)
        if rnd is None:
            continue
        cf = cfg.STRATEGIES_DIR / model / f"round{rnd}.py"
        if not cf.exists():
            continue
        try:
            agent = CodeAgent(cf.read_text(encoding="utf-8"),
                              n_stocks=cfg.N_STOCKS,
                              timeout=cfg.STRATEGY_TIMEOUT)
            res = run_backtest(agent, mkt, EnvConfig(initial_cash=cfg.INITIAL_CASH))
            ret = res.metrics["total_return"]
            eq = res.equity_curve
            results.append((model, rnd, ret, eq))
            print(f"  {model} R{rnd}: {ret*100:+.2f}%")
        except Exception as e:
            print(f"  {model}: 失败 {str(e)[:50]}")

    # 按收益排序后画图（收益从高到低）
    results.sort(key=lambda x: x[2], reverse=True)
    for model, rnd, ret, eq in results:
        st = STYLE.get(model, {"color": "gray", "lw": 2.0, "ls": "-"})
        label = f"{model} R{rnd} ({ret*100:+.2f}%)"
        ax.plot(eq / eq[0] * 100, label=label,
                color=st["color"], linewidth=st["lw"],
                linestyle=st["ls"], zorder=5, alpha=0.92)

    ax.set_xlabel("交易日", fontsize=15)
    ax.set_ylabel("净值（初始=100）", fontsize=15)
    ax.set_title(f"所有模型最佳轮次权益曲线 — 测试种子 {SEED}，5300股 × 252天",
                 fontsize=18, fontweight="bold", pad=15)
    ax.legend(loc="lower left", fontsize=10.5, framealpha=0.92, ncol=2)
    ax.grid(alpha=0.3, linestyle=":")
    ax.tick_params(labelsize=12)

    out = cfg.RESULTS_DIR.parent.parent / "figures" / "all_best_curves.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"\n✅ {out}")


if __name__ == "__main__":
    main()