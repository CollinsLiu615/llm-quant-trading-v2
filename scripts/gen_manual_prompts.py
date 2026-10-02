# scripts/gen_manual_prompts.py
"""生成给网页 AI 用的 prompt 文件，一次 2 个行业。
用法: python3 scripts/gen_manual_prompts.py deepseek
"""
import sys
from pathlib import Path

INDUSTRIES = ["新能源", "半导体", "医药", "白酒", "地产",
              "银行", "军工", "AI算力", "消费", "券商"]

TEMPLATE = """你是 A 股财经新闻编辑。请为以下 {n} 个行业各生成 64 条新闻标题：
{industries_str}

每个行业覆盖 8 种组合，每种 8 条：
- policy + 利好
- policy + 利空
- breaking + 利好
- breaking + 利空
- sentiment + 利好
- sentiment + 利空
- earnings + 利好
- earnings + 利空

要求：
- 标题 15~40 字，中文
- 提到具体公司名或具体政策/数据，避免"某公司""某行业"
- 风格模仿财联社/证券时报
- 每条附市场冲击估算：magnitude 0.005~0.10，decay 0.3~0.95，duration 3~30 天

只输出 JSON 数组，共 {total} 条，无 markdown 包裹、无解释。
每条格式：
{{"headline": "...", "magnitude": 0.025, "decay": 0.95, "duration": 20,
 "kind": "policy|breaking|sentiment|earnings",
 "industry": "{industry_options}",
 "direction": "利好|利空"}}
"""


def main():
    ai_name = sys.argv[1] if len(sys.argv) > 1 else "chatgpt"
    out_dir = Path(__file__).parent.parent / "data" / "manual_prompts"
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = [INDUSTRIES[i:i+2] for i in range(0, 10, 2)]

    lines = [f"# 给网页 AI 的 prompts（AI: {ai_name}）\n"]
    lines.append("## 操作流程\n")
    lines.append("1. 打开本文件，复制第 1 段 prompt")
    lines.append("2. 粘贴给网页 AI，等它输出 JSON")
    lines.append("3. **全选 AI 输出的 JSON 并复制**")
    lines.append("4. 终端跑下面这条命令保存：")
    lines.append(f"   `pbpaste > ~/Desktop/stock_ai_test_v2/data/news_raw/manual_{ai_name}_1.json`")
    lines.append(f"5. 段号 1→2→3→4→{len(pairs)}，重复以上步骤")
    lines.append(f"6. 全部完成后跑：`python3 scripts/merge_news.py`")
    lines.append("\n---\n")

    for i, pair in enumerate(pairs, 1):
        n = len(pair)
        total = n * 64
        industries_str = "  ".join(f"【{x}】" for x in pair)
        industry_options = "|".join(pair)
        prompt = TEMPLATE.format(
            n=n, industries_str=industries_str, total=total,
            industry_options=industry_options,
        )
        lines.append(f"\n## 第 {i} 段：{'+'.join(pair)}\n")
        lines.append("```")
        lines.append(prompt)
        lines.append("```\n")

    out_file = out_dir / f"{ai_name}_prompts.md"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 写入 {out_file}")
    print(f"\n下一步：open -e {out_file}")


if __name__ == "__main__":
    main()
