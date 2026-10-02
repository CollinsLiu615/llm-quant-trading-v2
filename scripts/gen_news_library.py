# scripts/gen_news_library.py
"""用 ECNU 生成 A 股新闻库。支持增量（已存在文件跳过）。"""
import os
import sys
import json
import time
import re
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from stock_sim_v4 import INDUSTRY_NAMES  # 只用到行业名

from openai import OpenAI

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "news_raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KINDS = ["policy", "breaking", "sentiment", "earnings"]
DIRECTIONS = ["利好", "利空"]

client = OpenAI(
    base_url="https://chat.ecnu.edu.cn/open/api/v1",
    api_key=os.environ["ECNU_API_KEY"],
    timeout=60.0,
)

PROMPT_TPL = """你是 A 股财经新闻编辑。请生成 8 条【{direction}】【{kind_cn}】类型的【{industry}】行业新闻标题。

要求：
- 标题 15~40 字，中文
- 提到具体公司名或具体政策/数据，避免"某公司""某行业"
- 风格模仿财联社/证券时报
- {direction}的新闻要明确体现利好/利空方向

每条附市场冲击估算：
- magnitude: 0.005 ~ 0.10（对数收益冲击）
- decay: 0.3 ~ 0.95（每日衰减）
- duration: 3 ~ 30 天

只输出 JSON 数组，无 markdown 包裹：
[{{"headline": "...", "magnitude": 0.025, "decay": 0.95, "duration": 20}}, ...]
"""

KIND_CN = {
    "policy": "政策",
    "breaking": "突发",
    "sentiment": "情绪",
    "earnings": "财报",
}


def parse_json_array(text: str):
    """从可能含 markdown 的响应里抽出 JSON 数组。"""
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def gen_one(kind, industry, direction, retries=2):
    fname = OUT_DIR / f"ecnu_{kind}_{industry}_{direction}.json"
    if fname.exists():
        print(f"  跳过（已存在）: {fname.name}")
        return
    prompt = PROMPT_TPL.format(kind_cn=KIND_CN[kind], industry=industry,
                               direction=direction)
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model="ecnu-max",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,   # 提高多样性
                max_tokens=2048,
            )
            text = resp.choices[0].message.content or ""
            arr = parse_json_array(text)
            if not arr or len(arr) == 0:
                print(f"  ⚠ 解析失败，重试 {attempt+1}")
                time.sleep(2)
                continue
            # 补全元信息
            for item in arr:
                item["kind"] = kind
                item["industry"] = industry
                item["direction"] = direction
                item["source"] = "ecnu"
            fname.write_text(json.dumps(arr, ensure_ascii=False, indent=2),
                             encoding="utf-8")
            print(f"  ✅ {fname.name} ({len(arr)} 条)")
            return
        except Exception as e:
            print(f"  ❌ {type(e).__name__}: {str(e)[:80]}")
            time.sleep(2)
    print(f"  ✗ 最终失败: {fname.name}")


def main():
    # 命令行过滤: python3 gen_news_library.py policy 新能源 利好
    if len(sys.argv) == 4:
        gen_one(sys.argv[1], sys.argv[2], sys.argv[3])
        return
    combos = list(product(KINDS, INDUSTRY_NAMES, DIRECTIONS))
    print(f"共 {len(combos)} 组待生成\n")
    for i, (k, ind, d) in enumerate(combos, 1):
        print(f"[{i}/{len(combos)}] {k} | {ind} | {d}")
        gen_one(k, ind, d)
        time.sleep(1)


if __name__ == "__main__":
    main()
