# scripts/final_summary.py
"""最终汇总：每模型最佳轮次 + 完整测试指标 + 排名。"""
import sys, json
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from pathlib import Path
import numpy as np
from code_agent import CodeAgent
from stock_sim_v4 import MarketConfig, MarketSimulator, EnvConfig, run_backtest
import contest_config as cfg
from models_v4 import MODELS_ORDERED

import sys as _sys
_sys.stdout.reconfigure(line_buffering=True)


# grok 手动补充（multi_round 启动时它不在列表）
GROK_EXTRA = {"round1": -0.2573, "round2": -0.1514}

# gpt 补充
GPT_EXTRA = {"round1": -0.2978, "round2": -0.1661}


def get_round_avg(model, rnd):
    """从 multi_round_summary.json 拿指定轮次的测试集平均收益。"""
    if model == "grok-4.7":
        return GROK_EXTRA.get(f"round{rnd}")
    if model == "gpt-5.6-sol":
        # gpt R3/R4 是 R2 复制的，用 R2 值
        if rnd >= 2:
            return GPT_EXTRA["round2"]
        return GPT_EXTRA.get(f"round{rnd}")
    summary_file = cfg.RESULTS_DIR / "multi_round_summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    d = summary.get(model, {}).get(f"round{rnd}")
    if d and "avg" in d:
        return d["avg"]
    return None


def main():
    # 预生成测试市场
    test_seeds = [s for g in cfg.TEST_SEED_GROUPS for s in g]
    print(f"预生成 {len(test_seeds)} 个测试市场...")
    markets = {}
    for s in test_seeds:
        markets[s] = MarketSimulator(
            MarketConfig(n_stocks=cfg.N_STOCKS, n_days=cfg.N_DAYS, seed=s)).generate()

    results = {}
    for model in MODELS_ORDERED:
        # 找最佳轮次（排除空仓=0%）
        rounds = {}
        for rnd in [1, 2, 3, 4]:
            avg = get_round_avg(model, rnd)
            if avg is not None and abs(avg) > 1e-4:  # 非空仓
                rounds[rnd] = avg

        if not rounds:
            results[model] = {
                "status": "all_failed",
                "best_round": None,
                "rounds": {},
            }
            print(f"  {model}: ❌ 全部空仓/失败")
            continue

        best_rnd = max(rounds, key=lambda r: rounds[r])  # 收益最高（越大越好）
        best_avg = rounds[best_rnd]

        # 检查稳定性：至少 2 轮有效
        status = "stable" if len(rounds) >= 2 else "unstable"

        # 跑最佳轮次完整指标
        code = (cfg.STRATEGIES_DIR / model / f"round{best_rnd}.py").read_text(encoding="utf-8")
        agent = CodeAgent(code, n_stocks=cfg.N_STOCKS, timeout=cfg.STRATEGY_TIMEOUT)
        metrics_list = []
        for s in test_seeds:
            res = run_backtest(agent, markets[s], EnvConfig(initial_cash=cfg.INITIAL_CASH))
            metrics_list.append(res.metrics)

        rets = [m["total_return"] for m in metrics_list]
        results[model] = {
            "status": status,
            "best_round": best_rnd,
            "rounds": rounds,
            "avg_return": float(np.mean(rets)),
            "sharpe": float(np.mean([m["sharpe"] for m in metrics_list])),
            "max_drawdown": float(np.min([m["max_drawdown"] for m in metrics_list])),
            "turnover": float(np.mean([m["turnover"] for m in metrics_list])),
            "variance": float(np.var(rets)),
            "group_avgs": [
                float(np.mean(rets[i*6:(i+1)*6])) for i in range(3)
            ],
        }
        print(f"  {model}: {best_avg*100:+.2f}% (R{best_rnd}, {status})")

    # 落盘
    out = cfg.RESULTS_DIR / "final_summary.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 写入 {out}")

    # 打印排名表
    print("\n=== 最终排名（最佳轮次） ===")
    print(f"{'排名':<4}{'模型':<28}{'最佳轮次':<8}{'平均收益':<12}{'夏普':<8}{'回撤':<10}{'状态'}")
    print("-" * 85)

    ranked = [(m, d) for m, d in results.items() if d.get("status") == "stable"]
    ranked.sort(key=lambda x: x[1]["avg_return"], reverse=True)

    for i, (model, d) in enumerate(ranked, 1):
        print(f"{i:<4}{model:<28}R{d['best_round']:<7}"
              f"{d['avg_return']*100:>+8.2f}%   "
              f"{d['sharpe']:>+5.2f}   "
              f"{d['max_drawdown']*100:>+6.1f}%   "
              f"{d['status']}")

    print("\n--- 不稳定（< 2 轮有效） ---")
    for model, d in results.items():
        if d.get("status") == "unstable":
            print(f"  {model}: R{d['best_round']} {d['avg_return']*100:+.2f}%")

    print("\n--- 全失败 ---")
    for model, d in results.items():
        if d.get("status") == "all_failed":
            print(f"  {model}")


if __name__ == "__main__":
    main()