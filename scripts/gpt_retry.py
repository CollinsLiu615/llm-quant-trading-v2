# scripts/gpt_retry.py
"""为 gpt-5.6-sol 从 R1 到 R4 完整跑一遍（替换 grok）。"""
import sys, os, time, json
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from openai import OpenAI
from code_agent import CodeAgent, extract_code, check_code_safety, SecurityError
from contest_prompts import INITIAL_PROMPT, build_feedback_prompt
from stock_sim_v4 import MarketConfig, MarketSimulator, EnvConfig, run_backtest
import contest_config as cfg


MODEL = "gpt-5.6-sol"
BASE_URL = "https://codecraftapi.com/v1"
MODEL_ID = "gpt-5.6-sol"
KEY_ENV = "CODECRAFT_API_KEY"


def call_once(prompt, max_wait=600.0):
    """流式 + 硬超时。"""
    import threading
    key = os.environ[KEY_ENV]
    client = OpenAI(base_url=BASE_URL, api_key=key, timeout=120.0)

    result = {"text": None, "err": None, "done": False}

    def _run():
        try:
            t0 = time.time()
            stream = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=16384,
                stream=True,
                stream_options={"include_usage": True},
            )
            chunks = []
            usage = None
            last_print = time.time()
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = chunk.usage
                # 每 30 秒打印进度
                if time.time() - last_print > 30:
                    print(f"    ...{time.time()-t0:.0f}s, {sum(len(c) for c in chunks)} 字符")
                    last_print = time.time()
            result["text"] = "".join(chunks)
            dt = time.time() - t0
            if usage:
                print(f"  API: {dt:.0f}s, tokens in={usage.prompt_tokens} out={usage.completion_tokens}")
            else:
                print(f"  API: {dt:.0f}s")
        except Exception as e:
            result["err"] = e
        finally:
            result["done"] = True

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=max_wait)
    if not result["done"]:
        raise TimeoutError(f"硬超时 {max_wait}s")
    if result["err"]:
        raise result["err"]
    return result["text"]


def gen_round(round_num):
    # 缓存跳过：代码和结果都已存在 → 直接跳过
    code_file = cfg.STRATEGIES_DIR / MODEL / f"round{round_num}.py"
    result_file = cfg.RESULTS_DIR / MODEL / f"round{round_num}.json"
    if code_file.exists() and result_file.exists():
        print(f"  ⏭ R{round_num} 已缓存，跳过")
        return True

    if round_num == 1:
        prompt = INITIAL_PROMPT
    else:
        prev_code_file = cfg.STRATEGIES_DIR / MODEL / f"round{round_num-1}.py"
        prev_result_file = cfg.RESULTS_DIR / MODEL / f"round{round_num-1}.json"
        if not prev_code_file.exists() or not prev_result_file.exists():
            print(f"  ❌ 无 R{round_num-1} 结果")
            return False
        prev_code = prev_code_file.read_text(encoding="utf-8")
        prev_data = json.loads(prev_result_file.read_text(encoding="utf-8"))
        prompt = build_feedback_prompt(
            prev_code, prev_data["train_metrics"], cfg.TRAIN_SEEDS)

    print(f"  → 生成 R{round_num}...")
    try:
        text = call_once(prompt)
    except Exception as e:
        print(f"  ❌ API: {type(e).__name__}: {str(e)[:120]}")
        return False

    code = extract_code(text)
    if code is None:
        print(f"  ❌ 无法提取代码")
        return False
    try:
        check_code_safety(code)
    except SecurityError as e:
        err_msg = str(e)
        print(f"  ⚠ 安全失败: {err_msg}")
        # 把错误反馈给 AI，让它重写
        retry_prompt = (prompt + f"\n\n【系统反馈】你上次的代码被拒：{err_msg}\n"
                        "请压缩代码到 150 行以内，去掉冗余注释，只保留核心逻辑。"
                        "只输出代码，不要解释。")
        print(f"  → 重试...")
        try:
            text = call_once(retry_prompt)
        except Exception as e2:
            print(f"  ❌ 重试 API 失败: {type(e2).__name__}")
            return False
        code = extract_code(text)
        if code is None:
            print(f"  ❌ 重试仍无法提取代码")
            return False
        try:
            check_code_safety(code)
        except SecurityError as e2:
            print(f"  ❌ 重试仍失败: {e2}")
            return False

    out = cfg.STRATEGIES_DIR / MODEL
    out.mkdir(parents=True, exist_ok=True)
    (out / f"round{round_num}.py").write_text(code, encoding="utf-8")
    print(f"  ✅ R{round_num} 代码 ({len(code)} 字符)")

    agent = CodeAgent(code, n_stocks=cfg.N_STOCKS, timeout=cfg.STRATEGY_TIMEOUT)
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
    print(f"  R{round_num} 训练集: {avg_ret*100:+.2f}%")

    result_file = cfg.RESULTS_DIR / MODEL / f"round{round_num}.json"
    result_file.parent.mkdir(parents=True, exist_ok=True)
    result_file.write_text(json.dumps({
        "train_metrics": train_metrics,
        "history": [{"status": "gpt_retry"}],
        "round": round_num,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


if __name__ == "__main__":
    print(f"=== gpt-5.6-sol 完整跑 4 轮 ===")
    for r in range(1, cfg.N_ROUNDS + 1):
        print(f"\n[Round {r}/{cfg.N_ROUNDS}]")
        ok = gen_round(r)
        if not ok:
            print(f"  Round {r} 失败，停止")
            break
    print("\n完成")
