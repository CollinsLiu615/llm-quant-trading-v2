# scripts/final_score.py
"""读 final_summary.json，按权重算总分排名。"""
import json
import sys
from pathlib import Path

sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
import contest_config as cfg


def rank_norm(values):
    """values: list of float, 越大越好。返回 [0, 1] 的排名分数，1 = 最好。"""
    n = len(values)
    if n <= 1:
        return [1.0] * n
    order = sorted(range(n), key=lambda i: values[i], reverse=True)
    ranks = [0.0] * n
    for position, idx in enumerate(order):
        ranks[idx] = (n - 1 - position) / (n - 1)
    return ranks


def main():
    f = cfg.RESULTS_DIR / "final_summary.json"
    data = json.loads(f.read_text(encoding="utf-8"))

    # 只保留 stable
    items = [(m, d) for m, d in data.items() if d.get("status") == "stable"]
    if not items:
        print("❌ 无 stable 模型")
        return

    models = [m for m, _ in items]
    avg_ret = [d["avg_return"] for _, d in items]
    sharpe = [d["sharpe"] for _, d in items]
    max_dd = [d["max_drawdown"] for _, d in items]           # 负值，越大越好
    turnover = [d["turnover"] for _, d in items]             # 越小越好
    variance = [d["variance"] for _, d in items]             # 越小越好

    r_ret = rank_norm(avg_ret)
    r_sharpe = rank_norm(sharpe)
    r_dd = rank_norm(max_dd)                                  # 已经是负值，越大越好
    r_turnover = rank_norm([-t for t in turnover])            # 取负，越小越好
    r_variance = rank_norm([-v for v in variance])            # 取负，越小越好

    W = {"ret": 50, "sharpe": 20, "dd": 15, "to": 10, "var": 5}

    results = []
    for i, m in enumerate(models):
        score = (W["ret"] * r_ret[i]
                 + W["sharpe"] * r_sharpe[i]
                 + W["dd"] * r_dd[i]
                 + W["to"] * r_turnover[i]
                 + W["var"] * r_variance[i])
        results.append({
            "model": m,
            "score": round(score, 2),
            "avg_return": avg_ret[i],
            "sharpe": sharpe[i],
            "max_drawdown": max_dd[i],
            "turnover": turnover[i],
            "variance": variance[i],
            "best_round": data[m]["best_round"],
            "rank_ret": round(r_ret[i], 3),
            "rank_sharpe": round(r_sharpe[i], 3),
            "rank_dd": round(r_dd[i], 3),
            "rank_to": round(r_turnover[i], 3),
            "rank_var": round(r_variance[i], 3),
        })

    results.sort(key=lambda x: x["score"], reverse=True)

    # 打印总分表
    print("\n=== 总分排名 ===")
    print(f"{'排名':<4}{'模型':<28}{'总分':<8}{'收益':<12}{'夏普':<8}{'回撤':<10}{'最佳轮'}")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        print(f"{i:<4}{r['model']:<28}"
              f"{r['score']:<8.2f}"
              f"{r['avg_return']*100:>+8.2f}%   "
              f"{r['sharpe']:>+5.2f}   "
              f"{r['max_drawdown']*100:>+6.1f}%   "
              f"R{r['best_round']}")

    # 分项排名
    print("\n=== 分项排名（rank，1.0=最好） ===")
    print(f"{'模型':<28}{'收益':<8}{'夏普':<8}{'回撤':<8}{'换手':<8}{'方差'}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<28}"
              f"{r['rank_ret']:<8.2f}"
              f"{r['rank_sharpe']:<8.2f}"
              f"{r['rank_dd']:<8.2f}"
              f"{r['rank_to']:<8.2f}"
              f"{r['rank_var']:.2f}")

    # 落盘
    out = cfg.RESULTS_DIR / "final_scores.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ 写入 {out}")


if __name__ == "__main__":
    main()