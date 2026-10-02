# scripts/probe_models.py
"""探活：对 MODELS 里每个模型发一次最小请求。不写日志、不计入评测。"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from stock_sim_v3 import MODELS


def probe_one(ep, timeout=30.0):
    key = os.environ.get(ep.api_key_env, "")
    if not key:
        return f"❌ 环境变量缺失: {ep.api_key_env}"
    try:
        from openai import OpenAI
        client = OpenAI(base_url=ep.base_url or None, api_key=key, timeout=timeout)
        t0 = time.time()
        resp = client.chat.completions.create(
            model=ep.model_id,
            messages=[{"role": "user", "content": "回复两个字：可用"}],
            max_tokens=16,
            temperature=0.0,
            **ep.extra,
        )
        dt = time.time() - t0
        content = (resp.choices[0].message.content or "")[:40]
        return f"✅ OK ({dt:.1f}s) → {content!r}"
    except Exception as e:
        return f"❌ FAIL: {type(e).__name__}: {str(e)[:120]}"


def main():
    print(f"共 {len(MODELS)} 个模型待探测\n")
    for name, ep in MODELS.items():
        print(f"[{name}] ", end="", flush=True)
        print(probe_one(ep))


if __name__ == "__main__":
    main()