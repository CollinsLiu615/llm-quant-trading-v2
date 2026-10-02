# scripts/merge_news.py
"""合并 data/news_raw/*.json → data/news_library.json
自动清理中文空格、补 source、按 kind 校准 magnitude。"""
import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "news_raw"
OUT_FILE = ROOT / "data" / "news_library.json"

# magnitude 按 kind 缩放（真实市场冲击没那么大）
MAG_SCALE = {
    "policy":    0.5,
    "breaking":  0.8,
    "sentiment": 0.7,
    "earnings":  0.9,
}
VALID_KINDS = set(MAG_SCALE.keys())
VALID_DIRECTIONS = {"利好", "利空"}


def clean_str(s):
    if not isinstance(s, str):
        return s
    return re.sub(r"\s+", "", s)


def parse_source(fname):
    stem = fname.stem
    if stem.startswith("ecnu_"):
        return "ecnu"
    m = re.match(r"manual_([a-zA-Z0-9]+)_", stem)
    return f"manual_{m.group(1)}" if m else "unknown"


def load_one(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"  ✗ 解析失败 {path.name}: {e}")
        return []
    if not isinstance(data, list):
        print(f"  ✗ 不是数组 {path.name}")
        return []
    source = parse_source(path)
    clean = []
    for item in data:
        if not isinstance(item, dict):
            continue
        for k in ("kind", "industry", "direction", "headline"):
            if k in item:
                item[k] = clean_str(item[k])
        if item.get("kind") not in VALID_KINDS: continue
        if item.get("direction") not in VALID_DIRECTIONS: continue
        if not item.get("headline"): continue
        try:
            item["magnitude"] = float(item.get("magnitude", 0.02))
            item["decay"] = float(item.get("decay", 0.85))
            item["duration"] = int(item.get("duration", 10))
        except Exception:
            continue
        item["magnitude"] = round(item["magnitude"] * MAG_SCALE[item["kind"]], 4)
        item["source"] = source
        clean.append(item)
    return clean


def main():
    files = sorted(RAW_DIR.glob("*.json"))
    print(f"找到 {len(files)} 个文件\n")
    all_news = []
    for f in files:
        items = load_one(f)
        if items:
            print(f"  ✅ {f.name}: {len(items)} 条")
            all_news.extend(items)
        else:
            print(f"  ⚠ {f.name}: 0 条")
    print(f"\n合并后总计: {len(all_news)} 条")
    print("\n--- 来源分布 ---")
    for k, v in Counter(x["source"] for x in all_news).most_common():
        print(f"  {k}: {v}")
    print("\n--- kind 分布 ---")
    for k, v in Counter(x["kind"] for x in all_news).most_common():
        print(f"  {k}: {v}")
    print("\n--- 行业分布 ---")
    for k, v in Counter(x["industry"] for x in all_news).most_common():
        print(f"  {k}: {v}")
    OUT_FILE.write_text(
        json.dumps(all_news, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 写入 {OUT_FILE}")


if __name__ == "__main__":
    main()
