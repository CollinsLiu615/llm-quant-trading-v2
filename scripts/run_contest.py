# scripts/run_contest.py
"""第二阶段主循环：4 轮生成-回测-反馈 + 最终测试。全自动。"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from models_v4 import MODELS_V4, MODELS_ORDERED
from contest_api import call_llm
from contest_prompts import INITIAL_PROMPT, build_feedback_prompt
from code_agent import CodeAgent, extract_code, check_code_safety, SecurityError
from stock_sim_v4 import MarketConfig, MarketSimulator, EnvConfig, run_backtest
import contest_config as cfg


_MARKET_CACHE = {}


def get_market(seed):
    """市场缓存：同一 seed 只生成一次，全局复用。"""
    if seed not in _MARKET_CACHE:
        mkt = MarketSimulator(MarketConfig(
            n_stocks=cfg.N_STOCKS, n_days=cfg.N_DAYS, seed=seed)).generate()
        _MARKET_CACHE[seed] = mkt
    return _MARKET_CACHE[seed]


def gen_code_with_retry(model_name, base_prompt):
    """生成代码，允许 1 次自动修 bug。
    返回 (code, history) —— code 为 None 表示最终失败。
    """
    prompt = base_prompt
    history = []
    for attempt in range(2):
        text, status = call_llm(model_name, prompt, log_dir=cfg.LOGS_DIR)
        if status != "ok":
            history.append({"attempt": attempt, "status": status})
            if attempt == 0:
                time.sleep(2)
                continue
            return None, history
        code = extract_code(text)
        if code is None:
            history.append({"attempt": attempt, "status": "extract_fail"})
            prompt = (base_prompt
                      + "\n\n【系统反馈】你上次没有输出代码块。"
                      + "请只输出用三个反引号包裹的 Python 代码。")
            continue
        try:
            check_code_safety(code)
            CodeAgent(code, n_stocks=cfg.N_STOCKS, timeout=cfg.STRATEGY_TIMEOUT)
            history.append({"attempt": attempt, "status": "ok"})
            return code, history
        except SecurityError as e:
            history.append({"attempt": attempt, "status": "safety_fail",
                            "err": str(e)[:200]})
            prompt = (base_prompt
                      + "\n\n【系统反馈】上次代码被拒: " + str(e)
                      + "\n请修正后只输出代码。")
            continue
    return None, history



def run_training_round(round_num):
    """并发版：API 并发，回测串行（CodeAgent 用 signal 必须在主线程）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    print(f"\n{'='*60}\nRound {round_num}/{cfg.N_ROUNDS}\n{'='*60}")

    # 1. 预生成市场（主线程）
    for seed in cfg.TRAIN_SEEDS:
        get_market(seed)

    # 2. 准备 prompt（主线程）
    tasks = []
    for model in MODELS_ORDERED:
        if round_num == 1:
            prompt = INITIAL_PROMPT
        else:
            prev_code_file = cfg.STRATEGIES_DIR / model / f"round{round_num-1}.py"
            prev_result_file = cfg.RESULTS_DIR / model / f"round{round_num-1}.json"
            if not prev_code_file.exists() or not prev_result_file.exists():
                print(f"  {model}: 无上轮结果，跳过")
                continue
            prev_code = prev_code_file.read_text(encoding="utf-8")
            prev_data = json.loads(prev_result_file.read_text(encoding="utf-8"))
            prompt = build_feedback_prompt(
                prev_code, prev_data["train_metrics"], cfg.TRAIN_SEEDS)
        tasks.append((model, prompt))

    # 3. 检查缓存：本轮代码文件已存在 → 直接读，跳过 API
    results = {}
    need_api = []
    for model, prompt in tasks:
        cached = cfg.STRATEGIES_DIR / model / f"round{round_num}.py"
        if cached.exists():
            code = cached.read_text(encoding="utf-8")
            results[model] = (code, [{"status": "cached"}])
            print(f"    ⏭ {model}: 复用缓存")
        else:
            need_api.append((model, prompt))

    # 4. 并发调 API（只对未缓存的模型）
    if need_api:
        print(f"  并发调用 {len(need_api)} 个模型 (workers=6)...")
        t_api = time.time()
        with ThreadPoolExecutor(max_workers=6) as ex:
            futures = {ex.submit(gen_code_with_retry, m, p): m
                       for m, p in need_api}
            for fut in as_completed(futures):
                model = futures[fut]
                try:
                    code, history = fut.result()
                    results[model] = (code, history)
                    print(f"    ✓ {model}: {'ok' if code else 'fail'}")
                except Exception as e:
                    results[model] = (None, [{"error": str(e)[:100]}])
                    print(f"    ✗ {model}: {type(e).__name__}")
        print(f"  API 耗时 {time.time()-t_api:.0f}s")
    else:
        print(f"  全部复用缓存，跳过 API")

    # 4. 串行回测（主线程，signal 安全）
    print(f"  回测中...")
    for model in MODELS_ORDERED:
        if model not in results or results[model][0] is None:
            continue
        code, history = results[model]
        model_dir = cfg.STRATEGIES_DIR / model
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / f"round{round_num}.py").write_text(code, encoding="utf-8")

        train_metrics = []
        for seed in cfg.TRAIN_SEEDS:
            mkt = get_market(seed)
            agent = CodeAgent(code, n_stocks=cfg.N_STOCKS,
                              timeout=cfg.STRATEGY_TIMEOUT)
            res = run_backtest(agent, mkt,
                               EnvConfig(initial_cash=cfg.INITIAL_CASH))
            m = res.metrics
            m["seed"] = seed
            m["n_errors"] = agent.n_errors
            m["n_calls"] = agent.n_calls
            m["last_error"] = agent.last_error
            train_metrics.append(m)

        avg_ret = float(np.mean([m["total_return"] for m in train_metrics]))
        print(f"    [{model}] 平均收益 {avg_ret*100:+.2f}%")

        result_dir = cfg.RESULTS_DIR / model
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / f"round{round_num}.json").write_text(
            json.dumps({"train_metrics": train_metrics, "history": history,
                        "round": round_num}, ensure_ascii=False, indent=2),
            encoding="utf-8")


def run_final_test():
    """最终测试：3 组 × 6 个测试种子。"""
    print(f"\n{'='*60}\n最终测试\n{'='*60}")
    for i, model in enumerate(MODELS_ORDERED, 1):
        final_code_file = cfg.STRATEGIES_DIR / model / f"round{cfg.N_ROUNDS}.py"
        if not final_code_file.exists():
            print(f"[{i:2d}/{len(MODELS_ORDERED)}] {model}: 无最终代码，跳过")
            continue
        code = final_code_file.read_text(encoding="utf-8")
        test_metrics = []
        for g_i, group in enumerate(cfg.TEST_SEED_GROUPS, 1):
            for seed in group:
                mkt = get_market(seed)
                agent = CodeAgent(code, n_stocks=cfg.N_STOCKS,
                                  timeout=cfg.STRATEGY_TIMEOUT)
                res = run_backtest(agent, mkt,
                                   EnvConfig(initial_cash=cfg.INITIAL_CASH))
                m = res.metrics
                m["seed"] = seed
                m["group"] = g_i
                test_metrics.append(m)

        avg_all = float(np.mean([m["total_return"] for m in test_metrics]))
        group_avgs = []
        for g_i in range(1, len(cfg.TEST_SEED_GROUPS) + 1):
            gm = [m["total_return"] for m in test_metrics if m["group"] == g_i]
            group_avgs.append(float(np.mean(gm)))

        grp_str = " / ".join(f"组{k+1} {v*100:+.1f}%"
                             for k, v in enumerate(group_avgs))
        print(f"[{i:2d}/{len(MODELS_ORDERED)}] {model}: "
              f"测试均值 {avg_all*100:+.2f}%  ({grp_str})")

        result_dir = cfg.RESULTS_DIR / model
        result_dir.mkdir(parents=True, exist_ok=True)
        (result_dir / "test.json").write_text(
            json.dumps({"test_metrics": test_metrics, "avg_return": avg_all,
                        "group_avgs": group_avgs},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")


def main():
    print("=" * 60)
    print("第二阶段：AI 量化策略大赛")
    print("=" * 60)
    print(f"股票 {cfg.N_STOCKS} 只 × {cfg.N_DAYS} 天 × {cfg.N_ROUNDS} 轮")
    print(f"训练种子: {cfg.TRAIN_SEEDS}")
    print(f"测试: {len(cfg.TEST_SEED_GROUPS)} 组 × "
          f"{len(cfg.TEST_SEED_GROUPS[0])} 个")
    print(f"模型: {len(MODELS_ORDERED)} 个")
    print(f"初始资金: {cfg.INITIAL_CASH:,.0f} 元")

    t_start = time.time()
    for r in range(1, cfg.N_ROUNDS + 1):
        run_training_round(r)
    run_final_test()
    print(f"\n总耗时: {(time.time()-t_start)/60:.1f} 分钟")
    print(f"结果目录: {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()