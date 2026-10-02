# scripts/demo_min.py
"""最小验证：持续持仓策略，不调 API。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from stock_sim_v3 import (
    MarketConfig, MarketSimulator, EnvConfig,
    Agent, run_backtest,
)


class BuyOnceAgent(Agent):
    """Day 1 建仓后维持不动（返回当前持仓比例，不产生新交易）。"""
    def __init__(self, n, weight=0.6):
        self.n = n
        self.w = weight
        self._built = False

    def reset(self):
        self._built = False

    def act(self, obs):
        if not self._built:
            self._built = True
            return np.full(self.n, self.w / self.n)
        # 之后的每一天：返回"当前持仓占总资产的比例"，等于不再调仓
        eq = obs["equity"]
        if eq <= 0:
            return np.zeros(self.n)
        cur_val = obs["shares"] * obs["last_prices"]
        return cur_val / eq


def main():
    n_stocks = 5
    n_days = 180      # ← 从 20 提到 180，让新闻系统触发
    seed = 42

    names = [f"股票{i}" for i in range(n_stocks)]
    market_cfg = MarketConfig(
        n_stocks=n_stocks, n_days=n_days, seed=seed,
        stock_names=names,
    )
    env_cfg = EnvConfig(initial_cash=1_000_000.0)

    market = MarketSimulator(market_cfg).generate()
    print(f"市场生成: {market.prices.shape}, 事件数={len(market.events)}")

    agent = BuyOnceAgent(n_stocks, weight=0.6)
    res = run_backtest(agent, market, env_cfg)

    m = res.metrics
    print(f"\n=== 回测结果 (5股 x {n_days}天 x 1种子) ===")
    print(f"总收益:     {m['total_return']:+.2%}")
    print(f"年化收益:   {m['annual_return']:+.2%}")
    print(f"波动率:     {m['volatility']:.2%}")
    print(f"夏普:       {m['sharpe']:.2f}")
    print(f"最大回撤:   {m['max_drawdown']:.2%}")
    print(f"胜率:       {m['win_rate']:.2%}")
    print(f"调仓次数:   {m['n_trades']:.0f}")
    print(f"累计成交:   {m['turnover']:.0f} 元")
    print(f"\n权益曲线首尾: {res.equity_curve[0]:.0f} → {res.equity_curve[-1]:.0f}")


if __name__ == "__main__":
    main()
