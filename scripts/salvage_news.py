# scripts/salvage_news.py
"""从截断的 JSON 里抢救完整的对象。
用括号计数法逐条扫描，遇到完整对象就收，遇到断尾就停。
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "news_raw"


def salvage(text):
    """从可能截断的 JSON 文本里提取完整的 {...} 对象列表。"""
    # 找到第一个 '[' 的位置
    start = text.find("[")
    if start < 0:
        return []
    text = text[start:]
    items = []
    depth = 0
    in_str = False
    escape = False
    buf_start = None
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "{":
            if depth == 0:
                buf_start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and buf_start is not None:
                chunk = text[buf_start:i + 1]
                try:
                    obj = json.loads(chunk)
                    items.append(obj)
                except Exception:
                    pass
                buf_start = None
        elif ch == "]" and depth == 0:
            break
    return items


def process(fname):
    p = RAW / fname
    text = p.read_text(encoding="utf-8")
    items = salvage(text)
    if not items:
        print(f"  ✗ {fname}: 抢救 0 条")
        return
    # 检查必需字段
    valid = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if not it.get("headline"):
            continue
        if it.get("kind") not in ("policy", "breaking", "sentiment", "earnings"):
            continue
        if it.get("direction") not in ("利好", "利空"):
            continue
        if not it.get("industry"):
            continue
        valid.append(it)
    # 覆盖原文件为纯净 JSON
    backup = p.with_suffix(".json.bak")
    if not backup.exists():
        backup.write_text(text, encoding="utf-8")
    p.write_text(json.dumps(valid, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✅ {fname}: 抢救 {len(valid)}/{len(items)} 条 (原始备份 {backup.name})")


def main():
    targets = sys.argv[1:] or []
    if not targets:
        targets = [f.name for f in sorted(RAW.glob("*.json"))
                   if "qwen" in f.name or "kimi" in f.name]
    print(f"待处理 {len(targets)} 个文件\n")
    for t in targets:
        process(t)


if __name__ == "__main__":
    main()
