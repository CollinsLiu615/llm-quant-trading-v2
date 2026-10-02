# scripts/describe_strategies.py
"""读每个模型最佳轮次的代码，跑一次回测，输出大白话总结。"""
import sys, json, os
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'src')
from pathlib import Path
import numpy as np
from code_agent import CodeAgent
from stock_sim_v4 import MarketConfig, MarketSimulator, EnvConfig, run_backtest, INDUSTRY_NAMES
import contest_config as cfg
from models_v4 import MODELS_ORDERED
from openai import OpenAI


SEED = 2513
ECNU_CLIENT = OpenAI(
    base_url="https://chat.ecnu.edu.cn/open/api/v1",
    api_key=os.environ.get("ECNU_API_KEY", ""),
    timeout=60.0,
)

# 从 final_scores.json 读最佳轮次
BEST_ROUNDS = {}
_scores = cfg.RESULTS_DIR / "final_scores.json"
if _scores.exists():
    for r in json.loads(_scores.read_text(encoding="utf-8")):
        BEST_ROUNDS[r["model"]] = r["best_round"]
BEST_ROUNDS.setdefault("Xing4.0-29B-A4B", 3)


def ask_ecnu(code, model_name):
    """让 ECNU 用大白话总结策略。"""
    prompt = f"""以下是一个 AI 写的股票交易策略代码。请用**普通人能懂的话**总结它：

【要求】
- 100 字以内
- 不要技术术语（不要提"夏普""因子"这些）
- 用"它主要做什么"、"什么时候买"、"什么时候卖"这种表达
- 语气客观

【代码】
{code[:3000]}

【输出格式】
直接一段话，不要前缀、不要 markdown。
"""
    try:
        resp = ECNU_CLIENT.chat.completions.create(
            model="ecnu-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"（ECNU 调用失败：{str(e)[:60]}）"


def analyze_behavior(agent, mkt):
    """跑一次回测，统计行为特征。"""
    res = run_backtest(agent, mkt, EnvConfig(initial_cash=cfg.INITIAL_CASH))
    env = res.env

    # 每天持仓数与仓位
    pos_counts = []
    positions = []
    daily_weights = []
    for t in range(1, len(env.equity_curve)):
        # 从 trades 里推不出每天持仓，需重新跟踪
        pass

    # 更简单：从最后一次 shares 和 equity_curve 反推
    # 遍历 trades 重建每日持仓
    share_hist = [np.zeros(env.n_stocks)]
    cur = np.zeros(env.n_stocks)
    for tr in env.trades:
        cur = cur + np.array(tr["buy"]) - np.array(tr["sell"])
        share_hist.append(cur.copy())

    eq_curve = np.asarray(env.equity_curve)
    n_days = min(len(share_hist), len(eq_curve))

    pos_counts = []
    pos_ratios = []
    industry_freq = np.zeros(len(INDUSTRY_NAMES))
    total_trades = len(env.trades)

    for t in range(n_days):
        shares = share_hist[t]
        n_hold = int((shares > 0).sum())
        pos_counts.append(n_hold)
        if n_hold > 0:
            # 该日持仓的行业分布
            held_inds = mkt.industry_id[shares > 0]
            for ind in held_inds:
                industry_freq[ind] += 1
        price = env.prices[min(t, env.n_days - 1)]
        pos_val = float(shares @ price)
        eq = eq_curve[t]
        pos_ratios.append(pos_val / eq if eq > 0 else 0.0)

    pos_counts = np.array(pos_counts)
    pos_ratios = np.array(pos_ratios)

    # 最常买的行业 Top3
    if industry_freq.sum() > 0:
        top3_idx = np.argsort(-industry_freq)[:3]
        top3 = [(INDUSTRY_NAMES[i], int(industry_freq[i]))
                for i in top3_idx if industry_freq[i] > 0]
    else:
        top3 = []

    return {
        "final_return": res.metrics["total_return"],
        "avg_hold": float(pos_counts.mean()),
        "max_hold": int(pos_counts.max()),
        "avg_position": float(pos_ratios.mean()),
        "max_position": float(pos_ratios.max()),
        "n_trades": total_trades,
        "top_industries": top3,
        "n_errors": agent.n_errors,
    }


def main():
    print(f"生成测试市场 seed={SEED}...")
    mkt = MarketSimulator(MarketConfig(n_stocks=cfg.N_STOCKS,
                                       n_days=cfg.N_DAYS, seed=SEED)).generate()

    out_lines = ["# 各模型策略「大白话」总结\n"]
    out_lines.append(f"（测试市场：seed={SEED}，5300 股 × 252 天）\n")

    for model in MODELS_ORDERED + ["Xing4.0-29B-A4B"]:
        rnd = BEST_ROUNDS.get(model)
        if rnd is None:
            continue
        code_file = cfg.STRATEGIES_DIR / model / f"round{rnd}.py"
        if not code_file.exists():
            print(f"  {model}: 无 R{rnd} 代码")
            continue
        code = code_file.read_text(encoding="utf-8")

        print(f"▶ {model} R{rnd} ...", end="", flush=True)
        try:
            agent = CodeAgent(code, n_stocks=cfg.N_STOCKS,
                              timeout=cfg.STRATEGY_TIMEOUT)
            beh = analyze_behavior(agent, mkt)
        except Exception as e:
            print(f" 失败：{str(e)[:60]}")
            beh = None

        summary = ask_ecnu(code, model)
        print(f" ok")

        out_lines.append(f"\n## {model}（最佳轮次 R{rnd}）\n")
        out_lines.append(f"**一句话总结**：{summary}\n")
        if beh:
            out_lines.append(f"\n**客观数据**：\n")
            out_lines.append(f"- 最终收益：{beh['final_return']*100:+.2f}%")
            out_lines.append(f"- 平均持仓 {beh['avg_hold']:.1f} 只（最多 {beh['max_hold']} 只）")
            out_lines.append(f"- 平均仓位 {beh['avg_position']*100:.1f}%（最高 {beh['max_position']*100:.1f}%）")
            out_lines.append(f"- 全年调仓 {beh['n_trades']} 次")
            if beh['top_industries']:
                inds = "、".join(f"{name}（{cnt}天次）" for name, cnt in beh['top_industries'])
                out_lines.append(f"- 最常买：{inds}")
            if beh['n_errors'] > 0:
                out_lines.append(f"- ⚠️ 运行错误 {beh['n_errors']}/{252} 天")

    out = cfg.RESULTS_DIR.parent.parent / "docs" / "策略大白话总结.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n✅ 写入 {out}")


if __name__ == "__main__":
    main()