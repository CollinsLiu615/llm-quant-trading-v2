import os, time
from openai import OpenAI
key = os.environ.get("CODECRAFT_API_KEY", "")
if not key:
    print("❌ 未设置 CODECRAFT_API_KEY"); raise SystemExit
client = OpenAI(base_url="https://codecraftapi.com/v1", api_key=key, timeout=60.0)
for name in ["grok-4.6", "seed-2.1-turbo"]:
    try:
        t0 = time.time()
        r = client.chat.completions.create(
            model=name,
            messages=[{"role": "user", "content": "回复两字：可用"}],
            max_tokens=32, temperature=0.0,
        )
        dt = time.time() - t0
        u = r.usage
        print(f"[{name}] ✅ {dt:.1f}s  tokens={u.total_tokens if u else '?'}  "
              f"返回: {(r.choices[0].message.content or '')[:30]!r}")
    except Exception as e:
        print(f"[{name}] ❌ {type(e).__name__}: {str(e)[:120]}")
    time.sleep(2)
