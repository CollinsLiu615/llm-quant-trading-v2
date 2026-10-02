# scripts/grok_retry.py
"""单独为 grok-4.7 生成 R3、R4 代码（加长超时，减小 max_tokens）。"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import json
import numpy as np
from openai import OpenAI
from code_agent import extract_code, check_code_safety, SecurityError
from contest_prompts import INITIAL_PROMPT, build_feedback_prompt
import contest_config as cfg


MODEL = "grok-4.7"
BASE_URL = "https://uuapi.io/v1"
MODEL_ID = "grok-4.7"
KEY_ENV = "UUAPI_GROK_KEY"


def call_grok(prompt, timeout=600.0, max_tokens=16384):
    """单次调用 grok，加长超时。"""
    import os
    key = os.environ[KEY_ENV]
    client = OpenAI(base_url=BASE_URL, api_key=key, timeout=timeout)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    dt = time.time() - t0
    text = resp.choices[0].message.content or ""
    u = resp.usage
    print(f"  调用完成: {dt:.0f}s, tokens={u.total_tokens if u else '?'}")
    return text


def gen_round(round_num):
    """生成指定轮次的代码。"""
    if round_num == 1:
        prompt = INITIAL_PROMPT
    else:
        prev_code_file = cfg.STRATEGIES_DIR / MODEL / f"round{round_num-1}.py"
        prev_result_file = cfg.RESULTS_DIR / MODEL / f"round{round_num-1}.json"
        if not prev_code_file.exists() or not prev_result_file.exists():
            print(f"  ❌ 无第 {round_num-1} 轮结果，无法生成 R{round_num}")
            return False
        prev_code = prev_code_file.read_text(encoding="utf-8")
        prev_data = json.loads(prev_result_file.read_text(encoding="utf-8"))
        prompt = build_feedback_prompt(
            prev_code, prev_data["train_metrics"], cfg.TRAIN_SEEDS)

    print(f"  → 调用 API 生成 R{round_num}...")
    try:
        text = call_grok(prompt)
    except Exception as e:
        print(f"  ❌ API 失败: {type(e).__name__}: {str(e)[:120]}")
        return False

    code = extract_code(text)
    if code is None:
        print(f"  ❌ 无法提取代码")
        return False
    try:
        check_code_safety(code)
    except SecurityError as e:
        print(f"  ❌ 代码不安全: {e}")
        return False

    out = cfg.STRATEGIES_DIR / MODEL
    out.mkdir(parents=True, exist_ok=True)
    (out / f"round{round_num}.py").write_text(code, encoding="utf-8")
    print(f"  ✅ R{round_num} 代码已保存 ({len(code)} 字符)")

    # 跑训练集回测 + 落结果
    from code_agent import CodeAgent
    from stock_sim_v4 import MarketConfig, MarketSimulator, EnvConfig, run_backtest
    agent = CodeAgent(code, n_stocks=cfg.N_STOCKS,
                      timeout=cfg.STRATEGY_TIMEOUT)
    train_metrics = []
    for seed in cfg.TRAIN_SEEDS:
        mkt = MarketSimulator(
            MarketConfig(n_stocks=cfg.N_STOCKS, n_days=cfg.N_DAYS, seed=seed)).generate()
        res = run_backtest(agent, mkt, EnvConfig(initial_cash=cfg.INITIAL_CASH))
        m = res.metrics
        m["seed"] = seed
        m["n_errors"] = agent.n_errors
        train_metrics.append(m)
    avg_ret = float(np.mean([m["total_return"] for m in train_metrics]))
    print(f"  R{round_num} 训练集平均收益: {avg_ret*100:+.2f}%")

    result_file = cfg.RESULTS_DIR / MODEL / f"round{round_num}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps({
        "train_metrics": train_metrics,
        "history": [{"status": "grok_retry"}],
        "round": round_num,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


if __name__ == "__main__":
    import sys as _sys
    round_num = int(_sys.argv[1]) if len(_sys.argv) > 1 else 3
    print(f"=== 单独重跑 grok-4.7 Round {round_num} ===")
    ok = gen_round(round_num)
    print(f"结果: {'成功' if ok else '失败'}")