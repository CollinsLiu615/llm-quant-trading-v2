# scripts/plot_per_model_curves.py
"""每个模型画一张图，只画它最佳轮次的权益曲线。"""
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

# 从 final_scores.json 读每模型最佳轮次
BEST_ROUNDS = {}
_scores = cfg.RESULTS_DIR / "final_scores.json"
if _scores.exists():
    for r in json.loads(_scores.read_text(encoding="utf-8")):
        BEST_ROUNDS[r["model"]] = r["best_round"]
# 补充不稳定/未排名的
for m, rnd in [("Xing4.0-29B-A4B", 3)]:
    BEST_ROUNDS.setdefault(m, rnd)


def main():
    print(f"生成市场 seed={SEED}...")
    mkt = MarketSimulator(MarketConfig(n_stocks=cfg.N_STOCKS,
                                       n_days=cfg.N_DAYS, seed=SEED)).generate()
    bench = mkt.prices.mean(axis=1)
    bench_norm = bench / bench[0] * 100

    out_dir = cfg.RESULTS_DIR.parent.parent / "figures" / "per_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in MODELS_ORDERED + ["Xing4.0-29B-A4B"]:
        rnd = BEST_ROUNDS.get(model)
        if rnd is None:
            print(f"  {model}: 无最佳轮次，跳过")
            continue
        f = cfg.STRATEGIES_DIR / model / f"round{rnd}.py"
        if not f.exists():
            print(f"  {model}: R{rnd} 文件不存在，跳过")
            continue

        fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
        # 市场基准
        ax.plot(bench_norm, color="black", linestyle="--",
                linewidth=2.5, label="市场均值", zorder=10, alpha=0.7)

        try:
            agent = CodeAgent(f.read_text(encoding="utf-8"),
                              n_stocks=cfg.N_STOCKS,
                              timeout=cfg.STRATEGY_TIMEOUT)
            res = run_backtest(agent, mkt,
                               EnvConfig(initial_cash=cfg.INITIAL_CASH))
            eq = res.equity_curve
            ret = res.metrics["total_return"]
            ax.plot(eq / eq[0] * 100,
                    color="#E63946", linewidth=2.8,
                    label=f"{model} R{rnd} ({ret*100:+.1f}%)",
                    zorder=5)
        except Exception as e:
            plt.close()
            print(f"  {model}: 回测失败 {str(e)[:50]}")
            continue

        ax.set_xlabel("交易日", fontsize=12)
        ax.set_ylabel("净值（初始=100）", fontsize=12)
        ax.set_title(f"{model} · 最佳轮次 R{rnd} · 测试种子 {SEED}",
                     fontsize=15, fontweight="bold")
        ax.legend(loc="best", fontsize=11)
        ax.grid(alpha=0.3, linestyle=":")
        ax.tick_params(labelsize=10)
        fig.tight_layout()

        out = out_dir / f"{model.replace('/', '_')}.png"
        fig.savefig(out)
        plt.close()
        print(f"✅ {out.name}")


if __name__ == "__main__":
    main()
