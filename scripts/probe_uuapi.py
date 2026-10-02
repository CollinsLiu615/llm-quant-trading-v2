# scripts/probe_uuapi.py
"""探活 uuapi.io 的两个模型。"""
import os, time
from openai import OpenAI

TESTS = [
    ("claude-opus-5", "UUAPI_CLAUDE_KEY"),
    ("grok-4.7",      "UUAPI_GROK_KEY"),
]

for mid, key_env in TESTS:
    key = os.environ.get(key_env, "")
    if not key:
        print(f"[{mid}] ❌ 环境变量缺失: {key_env}")
        continue
    client = OpenAI(base_url="https://uuapi.io/v1", api_key=key, timeout=60.0)
    try:
        t0 = time.time()
        r = client.chat.completions.create(
            model=mid,
            messages=[{"role": "user", "content": "回复两字：可用"}],
            max_tokens=32, temperature=0.0,
        )
        dt = time.time() - t0
        u = r.usage
        content = (r.choices[0].message.content or "")[:40]
        print(f"[{mid}] ✅ {dt:.1f}s  tokens={u.total_tokens if u else '?'}  返回: {content!r}")
    except Exception as e:
        print(f"[{mid}] ❌ {type(e).__name__}: {str(e)[:150]}")
    time.sleep(2)
