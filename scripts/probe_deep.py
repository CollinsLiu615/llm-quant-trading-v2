# scripts/probe_deep.py
"""深度探活：连发多次请求，测试模型是否在累计 token 后卡死。"""
import os
import sys
import time

CASES = {
    # ---- 天翼 AI ----
    "Xing4.0-29B-A4B": {
        "base_url": "https://openapi.teleagi.cn/aipaas/xing4-29b/v1",
        "model_id": "Xing4.0-29B-A4B",
        "api_key_env": "TELEAI_API_KEY",
        "param": "max_completion_tokens",
    },
    "TeleChat3-438B-A24B": {
        "base_url": "https://openapi.teleagi.cn/aipaas/telechat3-438b-a24b/v1",
        "model_id": "TeleChat3-438B-A24B",
        "api_key_env": "TELEAI_API_KEY",
        "param": "max_completion_tokens",
    },
    # ---- 硅基流动 ----
    "GLM-4-9B": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model_id": "THUDM/GLM-4-9B-0414",
        "api_key_env": "SILICONFLOW_API_KEY",
        "param": "max_tokens",
    },
    "GLM-Z1-9B": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model_id": "THUDM/GLM-Z1-9B-0414",
        "api_key_env": "SILICONFLOW_API_KEY",
        "param": "max_tokens",
    },
    "DeepSeek-R1-8B": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model_id": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        "api_key_env": "SILICONFLOW_API_KEY",
        "param": "max_tokens",
    },
    "Qwen3-8B": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model_id": "Qwen/Qwen3-8B",
        "api_key_env": "SILICONFLOW_API_KEY",
        "param": "max_tokens",
    },
}


def probe(name, cfg, n_calls=10, prompt_chars=4000, timeout=180.0):
    key = os.environ.get(cfg["api_key_env"], "")
    if not key:
        print(f"[{name}] ❌ 环境变量缺失 {cfg['api_key_env']}")
        return
    from openai import OpenAI
    client = OpenAI(base_url=cfg["base_url"], api_key=key, timeout=timeout)
    filler = "请分析以下市场数据并给出策略建议。" * (prompt_chars // 16)
    prompt = filler + "\n请用一句话回复：收到。"
    kwargs_extra = {cfg["param"]: 256}
    cumulative = 0
    for i in range(n_calls):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=cfg["model_id"],
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                **kwargs_extra,
            )
            dt = time.time() - t0
            content = resp.choices[0].message.content or ""
            u = resp.usage
            pt = u.prompt_tokens if u else 0
            ct = u.completion_tokens if u else 0
            cumulative += pt + ct
            print(f"[{name}] #{i+1:02d} ✅ {dt:5.1f}s  "
                  f"prompt={pt} 完成={ct} 累计={cumulative}  "
                  f"返回前30字: {content[:30]!r}")
        except Exception as e:
            dt = time.time() - t0
            print(f"[{name}] #{i+1:02d} ❌ {dt:5.1f}s  "
                  f"{type(e).__name__}: {str(e)[:100]}")
        time.sleep(1)


def main():
    names = sys.argv[1:] or list(CASES.keys())
    for name in names:
        if name not in CASES:
            print(f"[跳过] 未知模型 {name}")
            continue
        print(f"\n========== 测试 {name} ==========")
        probe(name, CASES[name])


if __name__ == "__main__":
    main()
