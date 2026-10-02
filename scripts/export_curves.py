# scripts/export_curves.py
"""导出所有模型最佳轮次的权益曲线到 JS 文件（供 HTML 加载）。"""
import sys, json
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from pathlib import Path
import numpy as np
from code_agent import CodeAgent
from stock_sim_v4 import MarketConfig, MarketSimulator, EnvConfig, run_backtest
import contest_config as cfg
from models_v4 import MODELS_ORDERED

SEED = 2513

COLORS = {
    "claude-opus-5":            "#E63946",
    "seed-2.1-turbo":           "#F77F00",
    "deepseek-v4-pro":          "#06A77D",
    "glm-5.3-flashx":           "#7209B7",
    "grok-4.7":                 "#FFB703",
    "gpt-5.6-sol":              "#1D3557",
    "deepseek-v4.1-flash":      "#9D0208",
    "DeepSeek-V4-Flash-0731":   "#457B9D",
    "Qwen3.8-27B":              "#2A9D8F",
    "DeepSeek-R1-8B":           "#8B5A2B",
    "TeleChat3-438B-A24B":      "#C77DFF",
    "Xing4.0-29B-A4B":          "#F72585",
}

best = {}
f = cfg.RESULTS_DIR / "final_scores.json"
if f.exists():
    for r in json.loads(f.read_text(encoding="utf-8")):
        best[r["model"]] = r["best_round"]
best["Xing4.0-29B-A4B"] = 3

print(f"生成市场 seed={SEED}...")
mkt = MarketSimulator(MarketConfig(n_stocks=cfg.N_STOCKS,
                                   n_days=cfg.N_DAYS, seed=SEED)).generate()
market_series = mkt.prices.mean(axis=1)
market_norm = (market_series / market_series[0] * 100).tolist()

data = {
    "seed": SEED,
    "n_days": cfg.N_DAYS,
    "market": market_norm,
    "curves": {},
}

for model in MODELS_ORDERED + ["Xing4.0-29B-A4B"]:
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
        eq = res.equity_curve
        eq_norm = (eq / eq[0] * 100).tolist()
        data["curves"][model] = {
            "round": rnd,
            "return": float(res.metrics["total_return"]),
            "sharpe": float(res.metrics["sharpe"]),
            "max_drawdown": float(res.metrics["max_drawdown"]),
            "color": COLORS.get(model, "#999999"),
            "equity": eq_norm,
        }
        print(f"  {model} R{rnd}: {res.metrics['total_return']*100:+.2f}%")
    except Exception as e:
        print(f"  {model}: 失败 {str(e)[:50]}")

out = cfg.RESULTS_DIR.parent.parent / "figures" / "curves_data.js"
out.write_text("window.CURVES_DATA = " + json.dumps(data, ensure_ascii=False) + ";",
               encoding="utf-8")
print(f"\n✅ {out}")