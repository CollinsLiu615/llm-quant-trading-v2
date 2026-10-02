# scripts/plot_learning_curves.py
"""画 4 张学习曲线（R1~R4 各自的测试集收益），2×2 拼图。"""
import json, sys
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "PingFang SC"]
rcParams["axes.unicode_minus"] = False
import contest_config as cfg


# 手动补充数据（不在 multi_round_summary 里的）
EXTRA = {
    "grok-4.7":    {1: -0.2573, 2: -0.1514, 3: -0.1514, 4: -0.1514},
    "gpt-5.6-sol": {1: -0.2978, 2: -0.1661, 3: -0.1661, 4: -0.1661},
}

# 颜色映射（12 个模型）
COLORS = {
    "seed-2.1-turbo":          "#E63946",
    "claude-opus-5":           "#F77F00",
    "deepseek-v4-pro":         "#06A77D",
    "grok-4.7":                "#7209B7",
    "glm-5.3-flashx":          "#1D3557",
    "gpt-5.6-sol":             "#FFB703",
    "deepseek-v4.1-flash":     "#9D0208",
    "DeepSeek-V4-Flash-0731":  "#457B9D",
    "Qwen3.8-27B":             "#2A9D8F",
    "DeepSeek-R1-8B":          "#8B5A2B",
    "TeleChat3-438B-A24B":     "#C77DFF",
    "Xing4.0-29B-A4B":         "#B0B0B0",
}


def load_data():
    multi = json.loads((cfg.RESULTS_DIR / "multi_round_summary.json").read_text(encoding="utf-8"))
    models = list(multi.keys())
    # 补 grok、gpt
    for m, vals in EXTRA.items():
        if m not in models:
            models.append(m)
        multi.setdefault(m, {})
        for rnd, v in vals.items():
            multi[m].setdefault(f"round{rnd}", {"avg": v})
    return multi, models


def main():
    data, models = load_data()

    fig, axes = plt.subplots(2, 2, figsize=(18, 12), dpi=150)
    axes = axes.flatten()

    for rnd_idx in range(4):
        ax = axes[rnd_idx]
        rnd = rnd_idx + 1
        # 收集数值
        xs, ys, labels, colors = [], [], [], []
        for i, m in enumerate(models):
            d = data.get(m, {}).get(f"round{rnd}")
            if d is None or "avg" not in d:
                continue
            xs.append(i)
            ys.append(d["avg"] * 100)
            labels.append(m)
            colors.append(COLORS.get(m, "gray"))

        # 柱状图
        bars = ax.bar(range(len(labels)), ys, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.set_ylabel("测试集平均收益 (%)", fontsize=11)
        ax.set_title(f"Round {rnd}", fontsize=16, fontweight="bold")
        ax.tick_params(axis="y", labelsize=10)

        # 数值标注
        for bar, v in zip(bars, ys):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h - 1.5 if h < 0 else h + 0.5,
                    f"{v:.1f}", ha="center", va="top" if h < 0 else "bottom",
                    fontsize=8, color="white" if h < 0 else "black",
                    fontweight="bold")

    fig.suptitle("AI 策略 4 轮迭代学习曲线（测试集 18 种子平均收益）",
                 fontsize=20, fontweight="bold", y=0.995)
    out = cfg.RESULTS_DIR.parent.parent / "figures" / "learning_curves.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"✅ {out}")


if __name__ == "__main__":
    main()