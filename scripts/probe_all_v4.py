# scripts/probe_all_v4.py
"""对 models_v4 里 14 个模型做连通性测试。"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from models_v4 import MODELS_V4, MODELS_ORDERED

from openai import OpenAI


def probe(ep, timeout=60.0):
    key = os.environ.get(ep.api_key_env, "")
    if not key:
        return f"❌ 环境变量缺失: {ep.api_key_env}"
    try:
        client = OpenAI(base_url=ep.base_url, api_key=key, timeout=timeout)
        # 组装参数：天翼用 max_completion_tokens，其他用 max_tokens
        kwargs = dict(ep.extra) if ep.extra else {}
        if "max_completion_tokens" in kwargs:
            kwargs["max_completion_tokens"] = 32
        elif "max_tokens" in kwargs:
            kwargs["max_tokens"] = 32
        else:
            kwargs["max_tokens"] = 32

        t0 = time.time()
        resp = client.chat.completions.create(
            model=ep.model_id,
            messages=[{"role": "user", "content": "回复两字：可用"}],
            temperature=0.0,
            **kwargs,
        )
        dt = time.time() - t0
        u = resp.usage
        content = (resp.choices[0].message.content or "")[:40]
        tokens = u.total_tokens if u else "?"
        return f"✅ {dt:5.1f}s  tokens={tokens}  返回={content!r}"
    except Exception as e:
        return f"❌ {type(e).__name__}: {str(e)[:100]}"


def main():
    print(f"=== 探活 {len(MODELS_ORDERED)} 个模型 ===\n")
    results = []
    for i, name in enumerate(MODELS_ORDERED, 1):
        ep = MODELS_V4[name]
        print(f"[{i:2d}/{len(MODELS_ORDERED)}] {name:24s} ", end="", flush=True)
        r = probe(ep)
        print(r)
        results.append((name, r))
        time.sleep(1)

    print("\n=== 汇总 ===")
    ok = [n for n, r in results if r.startswith("✅")]
    fail = [n for n, r in results if not r.startswith("✅")]
    print(f"通过: {len(ok)}/{len(results)}")
    if fail:
        print("失败:")
        for n in fail:
            print(f"  ✗ {n}")


if __name__ == "__main__":
    main()