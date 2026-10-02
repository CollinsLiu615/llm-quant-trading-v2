# scripts/preview_curves.py
"""预览：几个模型在同一种子下的权益曲线对比。"""
import sys
import time
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

from contest_api import call_llm_for_code
from contest_prompts import INITIAL_PROMPT
from code_agent import CodeAgent, SecurityError
from stock_sim_v4 import MarketConfig, MarketSimulator, EnvConfig, run_backtest


MODELS = ["claude-opus-5", "Qwen3.8-27B", "DeepSeek-V4-Flash-0731", "seed-2.1-turbo"]
SEED = 390
N_STOCKS = 5300


def main():
    print(f"市场：{N_STOCKS} 股 × 252 天 × seed={SEED}")
    mkt = MarketSimulator(MarketConfig(n_stocks=N_STOCKS, n_days=252, seed=SEED)).generate()
    base = (mkt.prices[-1] / mkt.prices[0] - 1)
    print(f"市场整体：中位 {np.median(base)*100:+.1f}%  算术均值 {base.mean()*100:+.1f}%")
    print()

    curves = {}
    for model in MODELS:
        print(f"▶ {model} ...", end="", flush=True)
        t0 = time.time()
        code, status, err = call_llm_for_code(model, INITIAL_PROMPT, n_stocks=N_STOCKS)
        if status != "ok":
            print(f" 失败 ({status}: {err[:60]})")
            continue
        try:
            agent = CodeAgent(code, n_stocks=N_STOCKS, timeout=5.0)
        except SecurityError as e:
            print(f" 编译失败 ({str(e)[:60]})")
            continue
        res = run_backtest(agent, mkt, EnvConfig(initial_cash=1_000_000))
        dt = time.time() - t0
        m = res.metrics
        print(f" 收益 {m['total_return']*100:+.2f}%  错误 {agent.n_errors}/{agent.n_calls}  ({dt:.0f}s)")
        curves[model] = res.equity_curve

    if not curves:
        print("\n⚠ 没有可画的曲线")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for model, eq in curves.items():
        eq_norm = eq / eq[0] * 100
        ax.plot(eq_norm, label=model, linewidth=1.5)
    bench = mkt.prices.mean(axis=1)
    bench_norm = bench / bench[0] * 100
    ax.plot(bench_norm, label="市场均值", color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("交易日")
    ax.set_ylabel("净值（初始=100）")
    ax.set_title(f"模型权益曲线对比 — seed={SEED}, {N_STOCKS}股 × 252天")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    out = ROOT / "figures" / "preview_curves.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print(f"\n✅ 图保存到: {out}")


if __name__ == "__main__":
    main()