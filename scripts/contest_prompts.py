# scripts/contest_prompts.py
"""给 AI 的任务 prompt + 反馈 prompt + 自动诊断引擎。"""
from typing import Dict, List

TICK = chr(96) * 3           # 三个反引号
TICK_PY = TICK + "python"

INITIAL_PROMPT = (
    "你是一个 A 股量化研究员。请设计一个交易策略，用 Python 实现。\n\n"
    "【市场规则】\n"
    "- 标的：5300 只 A 股股票，含主板(±10%)、创业板(±20%)、科创板(±20%)、北交所(±30%)、ST(±5%)\n"
    "- T+1：当日买入次日才能卖出\n"
    "- 印花税：万分之五（仅卖出）\n"
    "- 佣金：万分之三，最低 5 元\n"
    "- 过户费：十万分之一\n"
    "- 最小交易单位：100 股（一手）\n"
    "- 初始资金：1,000,000 元\n\n"
    "【账户约束】\n"
    "- 每日调仓一次，最多持仓 20 只\n"
    "- 单只股票权重 <= 20%\n"
    "- 只允许做多，禁止做空\n\n"
    "【可用数据】\n"
    "- prices: np.ndarray, shape (60, 5300)。**第 0 维是时间（60 天），第 1 维是股票（5300 只）**。\n"
    "- news: List[str]，近 5 天新闻标题\n"
    "- position: np.ndarray, shape (5300,)，当前持仓股数\n"
    "- cash: float，可用现金\n\n"
    "【任务】\n"
    "实现函数 strategy(prices, news, position, cash)，返回目标权重 "
    "np.ndarray, shape (5300,)，范围 [0, 1]，sum <= 1。\n\n"
    "【约束】\n"
    "- 只用 numpy（已导入为 np）+ math + 内建函数\n"
    "- 禁止 import，禁止文件读写、网络\n"
    "- 单次执行 < 5 秒，代码 < 150 行\n"
    "- 禁止双下划线属性\n\n"
    "\n"
    "【示例代码骨架】（注意维度顺序）\n"
    "```\n"
    "import numpy as np\n"
    "def strategy(prices, news, position, cash):\n"
    "    n_days, n_stocks = prices.shape  # 60, 5300\n"
    "    w = np.zeros(n_stocks)  # 返回 5300 维\n"
    "    w[:20] = 0.03          # 举例：前 20 只等权 3%\n"
    "    return w\n"
    "```\n"
    "【输出格式】\n"
    f"只输出 Python 代码，用 {TICK_PY} ... {TICK} 包裹。不要输出解释。"
)


FEEDBACK_TEMPLATE = (
    "【你的上一轮代码】\n"
    + TICK_PY + "\n"
    + "{prev_code}\n"
    + TICK + "\n\n"
    + "【本轮回测结果 — 6 个训练种子】\n"
    + "{results_table}\n\n"
    + "【汇总】\n"
    + "平均收益: {avg_return:+.2f}%  中位收益: {median_return:+.2f}%\n"
    + "平均夏普: {avg_sharpe:+.2f}  平均最大回撤: {avg_dd:+.2f}%\n"
    + "平均换手率: {avg_turnover:.1f}%  收益方差: {return_std:.2f}\n\n"
    + "【自动诊断】\n"
    + "{diagnosis}\n\n"
    + "【任务】\n"
    + "基于以上反馈，优化策略，重点解决诊断指出的问题。\n"
    + "输出完整新代码，用 " + TICK_PY + " ... " + TICK + " 包裹。只输出代码，不要解释。\n"
)


def diagnose(results: List[Dict]) -> str:
    if not results:
        return "（无数据）"

    # 优先级 1：代码报错
    total_errors = sum(int(r.get("n_errors", 0)) for r in results)
    if total_errors > len(results) * 20:   # 平均每种子 >20 天报错
        return ("【严重】你的代码在执行中大量报错（平均每种子报错 "
                f"{total_errors//len(results)} 天）。"
                "很可能原因：\n"
                "1. 数组维度不匹配（如 prices[-5:] / prices[-6:] 长度不同）\n"
                "2. 索引越界（如 prices[-21] 在只有 60 行时也 OK，但要注意切片长度）\n"
                "3. 使用了 news 作为二维数组（news 是 List[str]，不是矩阵）\n"
                "请仔细检查每行 numpy 运算的输入形状，确保一致。"
                "返回的 weights 必须是长度 5300 的一维数组。")
    returns = [r["total_return"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    dds = [r["max_drawdown"] for r in results]
    turnovers = [r.get("turnover", 0) for r in results]
    avg_ret = sum(returns) / len(returns)
    std_ret = (sum((x - avg_ret) ** 2 for x in returns) / len(returns)) ** 0.5
    worst_i = min(range(len(returns)), key=lambda i: returns[i])
    n_loss = sum(1 for r in returns if r < 0)
    lines = []
    if n_loss > 0:
        lines.append(f"· {n_loss}/{len(returns)} 个种子亏损，最差 {returns[worst_i]*100:+.1f}%。推测：缺少熊市防守。")
    else:
        lines.append("· 所有种子盈利，抗跌性良好。")
    if std_ret > 0.20:
        lines.append(f"· 收益方差大（{std_ret*100:.1f}%），对市场状态敏感。")
    avg_to = sum(turnovers) / len(turnovers) / 1_000_000 if turnovers else 0
    if avg_to > 5.0:
        lines.append(f"· 换手率偏高（年均 {avg_to:.1f} 倍资金）。")
    if sum(sharpes) / len(sharpes) < 0:
        lines.append("· 平均夏普为负，风险调整后收益差。")
    if min(dds) < -0.30:
        lines.append(f"· 最大回撤 {min(dds)*100:+.1f}%，风控不足。")
    lines.append("")
    lines.append("【优化方向建议】")
    if n_loss > 0: lines.append("1. 加入市场状态识别，熊市降仓")
    if avg_to > 5.0: lines.append("2. 减少无谓调仓")
    if min(dds) < -0.30: lines.append("3. 增加止损")
    if std_ret > 0.20: lines.append("4. 增加自适应")
    return "\n".join(lines)


def format_results_table(results: List[Dict], seeds: List[int]) -> str:
    lines = ["种子       收益      夏普    最大回撤   换手"]
    for seed, r in zip(seeds, results):
        lines.append(
            f"{seed:>6}   {r['total_return']*100:>+7.2f}%  "
            f"{r['sharpe']:>+6.2f}  {r['max_drawdown']*100:>+7.2f}%  "
            f"{r.get('turnover', 0)/1_000_000:>6.1f}x"
        )
    return "\n".join(lines)


def build_feedback_prompt(prev_code: str, results: List[Dict], seeds: List[int]) -> str:
    returns = [r["total_return"] for r in results]
    sharpes = [r["sharpe"] for r in results]
    dds = [r["max_drawdown"] for r in results]
    turnovers = [r.get("turnover", 0) for r in results]
    avg_ret = sum(returns) / len(returns)
    median_ret = sorted(returns)[len(returns) // 2]
    avg_sharpe = sum(sharpes) / len(sharpes)
    avg_dd = sum(dds) / len(dds)
    avg_to = sum(turnovers) / len(turnovers) / 1_000_000
    std_ret = (sum((x - avg_ret) ** 2 for x in returns) / len(returns)) ** 0.5
    return FEEDBACK_TEMPLATE.format(
        prev_code=prev_code.strip(),
        results_table=format_results_table(results, seeds),
        avg_return=avg_ret * 100,
        median_return=median_ret * 100,
        avg_sharpe=avg_sharpe,
        avg_dd=avg_dd * 100,
        avg_turnover=avg_to * 100,
        return_std=std_ret * 100,
        diagnosis=diagnose(results),
    )