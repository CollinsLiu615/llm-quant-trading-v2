# scripts/plot_score_tables.py
"""把总分排名表和分项排名表画成图。"""
import json, sys
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "PingFang SC"]
rcParams["axes.unicode_minus"] = False
import contest_config as cfg


def style_cell(cell, highlight=False):
    cell.set_edgecolor("#cccccc")
    cell.set_linewidth(0.5)
    if highlight:
        cell.set_facecolor("#FFF3CD")
    else:
        cell.set_facecolor("#ffffff")


def main():
    data = json.loads((cfg.RESULTS_DIR / "final_scores.json").read_text(encoding="utf-8"))

    # ========== 表1：总分排名 ==========
    fig, ax = plt.subplots(figsize=(17, 8), dpi=150)
    ax.axis("off")
    ax.set_title("第二阶段 AI 量化策略大赛 · 总分排名",
                 fontsize=20, fontweight="bold", pad=25)

    headers = ["排名", "模型", "总分", "平均收益", "夏普", "最大回撤", "换手率", "方差", "最佳轮次"]
    rows = []
    for i, r in enumerate(data, 1):
        # 换手率：累计成交额 ÷ 初始资金（倍）
        turn_x = r["turnover"] / 10_000_000
        # 方差用百分比表示
        var_pct = r["variance"] ** 0.5 * 100  # 标准差更直观
        rows.append([
            str(i),
            r["model"],
            f"{r['score']:.1f}",
            f"{r['avg_return']*100:+.2f}%",
            f"{r['sharpe']:+.2f}",
            f"{r['max_drawdown']*100:+.2f}%",
            f"{turn_x:.1f}x",
            f"{var_pct:.1f}%",
            f"R{r['best_round']}",
        ])

    col_widths = [0.05, 0.24, 0.08, 0.11, 0.08, 0.11, 0.09, 0.09, 0.09]
    table = ax.table(cellText=rows, colLabels=headers,
                     colWidths=col_widths,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.2)

    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#2C3E50")
    # 前三名高亮
    for i in range(1, 4):
        for j in range(len(headers)):
            style_cell(table[(i, j)], highlight=True)
    # 其余
    for i in range(4, len(rows) + 1):
        for j in range(len(headers)):
            style_cell(table[(i, j)])

    # 底部备注
    fig.text(0.5, 0.02,
             "注：Xing4.0-29B-A4B 因 4 轮中仅 1 轮生成有效代码（R3 = -10.43%），"
             "归入「不稳定」档，不参与主排名。",
             ha="center", fontsize=11, color="#666666", style="italic")

    out1 = cfg.RESULTS_DIR.parent.parent / "figures" / "score_ranking.png"
    out1.parent.mkdir(exist_ok=True)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out1, bbox_inches="tight")
    print(f"✅ {out1}")

    # ========== 表2：分项排名 ==========
    fig, ax = plt.subplots(figsize=(13, 7), dpi=150)
    ax.axis("off")
    ax.set_title("分项排名（1.0 = 最好）", fontsize=20,
                 fontweight="bold", pad=25)

    headers = ["模型", "收益", "夏普", "回撤", "换手", "方差"]
    rows = []
    for r in data:
        rows.append([
            r["model"],
            f"{r['rank_ret']:.2f}",
            f"{r['rank_sharpe']:.2f}",
            f"{r['rank_dd']:.2f}",
            f"{r['rank_to']:.2f}",
            f"{r['rank_var']:.2f}",
        ])

    col_widths = [0.32, 0.13, 0.13, 0.13, 0.13, 0.13]
    table = ax.table(cellText=rows, colLabels=headers,
                     colWidths=col_widths,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.0)

    for j in range(len(headers)):
        cell = table[(0, j)]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("#2C3E50")
    # 颜色渐变（绿→红）
    import matplotlib.cm as cm
    for i, r in enumerate(data, start=1):
        vals = [r['rank_ret'], r['rank_sharpe'], r['rank_dd'],
                r['rank_to'], r['rank_var']]
        for j, v in enumerate(vals, start=1):
            # v 从 0 到 1，映射到红→绿
            color = cm.RdYlGn(v)
            table[(i, j)].set_facecolor(color)
            table[(i, j)].set_edgecolor("#cccccc")
            table[(i, j)].set_linewidth(0.5)
        table[(i, 0)].set_facecolor("#f8f8f8")
        table[(i, 0)].set_edgecolor("#cccccc")
        table[(i, 0)].set_linewidth(0.5)

    fig.text(0.5, 0.02,
             "注：Xing4.0-29B-A4B 因 4 轮中仅 1 轮生成有效代码，"
             "归入「不稳定」档，不参与分项排名。",
             ha="center", fontsize=11, color="#666666", style="italic")

    out2 = cfg.RESULTS_DIR.parent.parent / "figures" / "score_breakdown.png"
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out2, bbox_inches="tight")
    print(f"✅ {out2}")


if __name__ == "__main__":
    main()