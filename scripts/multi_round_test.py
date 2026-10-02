# scripts/multi_round_test.py
"""对每个模型的 round1~round4 代码，在测试集上回测，输出学习曲线。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from code_agent import CodeAgent
from stock_sim_v4 import MarketConfig, MarketSimulator, EnvConfig, run_backtest
import contest_config as cfg
from models_v4 import MODELS_ORDERED


def main():
    test_seeds = [s for g in cfg.TEST_SEED_GROUPS for s in g]
    print(f"预生成 {len(test_seeds)} 个测试市场...")
    markets = {}
    for s in test_seeds:
        markets[s] = MarketSimulator(
            MarketConfig(n_stocks=cfg.N_STOCKS, n_days=cfg.N_DAYS, seed=s)).generate()
    print("市场生成完毕\n")

    results = {}
    for model in MODELS_ORDERED:
        results[model] = {}
        for rnd in range(1, cfg.N_ROUNDS + 1):
            code_file = cfg.STRATEGIES_DIR / model / f"round{rnd}.py"
            if not code_file.exists():
                results[model][f"round{rnd}"] = None
                continue
            code = code_file.read_text(encoding="utf-8")
            try:
                agent = CodeAgent(code, n_stocks=cfg.N_STOCKS,
                                  timeout=cfg.STRATEGY_TIMEOUT)
            except Exception as e:
                results[model][f"round{rnd}"] = {"error": str(e)[:80]}
                continue
            rets = []
            for s in test_seeds:
                res = run_backtest(agent, markets[s],
                                   EnvConfig(initial_cash=cfg.INITIAL_CASH))
                rets.append(res.metrics["total_return"])
            results[model][f"round{rnd}"] = {
                "avg": float(np.mean(rets)),
                "group_avgs": [
                    float(np.mean(rets[i*6:(i+1)*6])) for i in range(3)
                ],
            }
            print(f"  {model} round{rnd}: "
                  f"{results[model][f'round{rnd}']['avg']*100:+.2f}%")

    out = ROOT / "data" / "contest_results" / "multi_round_summary.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 写入 {out}")

    print("\n=== 学习曲线 ===")
    header = "模型".ljust(28) + "R1".ljust(12) + "R2".ljust(12) + "R3".ljust(12) + "R4"
    print(header)
    print("-" * 80)
    for model in MODELS_ORDERED:
        row = model.ljust(28)
        for r in range(1, 5):
            d = results[model].get(f"round{r}")
            if d is None:
                row += "N/A".ljust(12)
            elif "error" in d:
                row += "ERR".ljust(12)
            else:
                row += f"{d['avg']*100:+.2f}%".ljust(12)
        print(row)


if __name__ == "__main__":
    main()