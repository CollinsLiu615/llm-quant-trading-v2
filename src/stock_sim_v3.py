# -*- coding: utf-8 -*-
"""
stock_sim.py v2 — 股票模拟器 + 市场情绪/新闻/政策 + 多模型 AI 评测

新增：
  - NewsEvent / NewsGenerator：政策、突发、情绪、财报四类事件
  - 事件作为收益率冲击注入行情，指数衰减
  - obs["news"] 把近期新闻文本喂给 Agent
  - RemoteAgent：任何 OpenAI 兼容端点的通用客户端
  - ModelEndpoint 注册表：集中管理各参赛模型
"""

from __future__ import annotations

import abc
import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# ============================================================
# 1. 新闻 / 情绪 / 政策事件
# ============================================================

@dataclass
class NewsEvent:
    day: int                    # 发生日（交易日索引）
    kind: str                   # 'policy' | 'breaking' | 'sentiment' | 'earnings'
    headline: str               # 新闻文本（喂给 AI）
    impacts: np.ndarray         # (n_stocks,) 当日对数收益冲击
    decay: float                # 每日衰减系数 0~1
    duration: int               # 持续天数


# 各类事件的模板（可自行扩充，会随机组合）
_POLICY_TPL = [
    ("利好", "国务院出台{industry}产业扶持政策，{stock}直接受益"),
    ("利空", "监管收紧{industry}行业，{stock}面临合规成本上升"),
    ("利好", "央行降准0.5个百分点，流动性宽松利好{stock}"),
    ("利空", "央行超预期加息，{stock}估值承压"),
    ("利好", "{industry}行业纳入新基建重点，{stock}订单预期上修"),
    ("利空", "环保督查进驻{industry}，{stock}部分产能停产"),
]
_BREAKING_TPL = [
    ("利好", "{stock}发布超预期季度业绩，营收同比增长{v}%"),
    ("利空", "{stock}被曝财务造假，证监会立案调查"),
    ("利好", "{stock}获海外大单，合同金额{v}亿元"),
    ("利空", "{stock}核心高管集体离职，市场信心受挫"),
    ("利好", "{stock}新产品发布引发市场热捧"),
    ("利空", "{stock}遭机构下调评级至“卖出”"),
]
_SENTIMENT_TPL = [
    ("利好", "北向资金大幅净流入{v}亿，市场情绪转暖"),
    ("利空", "外围市场暴跌，A股避险情绪升温"),
    ("利好", "两市成交额突破{v}万亿，做多情绪高涨"),
    ("利空", "市场传言IPO扩容，投资者观望情绪浓厚"),
]
_EARNINGS_TPL = [
    ("利好", "{stock}公布财报：净利润同比+{v}%，超市场预期"),
    ("利空", "{stock}公布财报：净利润同比-{v}%，不及预期"),
]

_INDUSTRIES = ["新能源", "半导体", "医药", "白酒", "地产", "银行", "军工", "AI算力"]


class NewsGenerator:
    """在每个交易日按概率生成新闻事件"""

    def __init__(self, n_stocks: int, n_days: int, rng: np.random.Generator,
                 stock_names: Optional[List[str]] = None,
                 p_policy: float = 0.015,
                 p_breaking: float = 0.02,
                 p_sentiment: float = 0.03,
                 p_earnings: float = 0.01):
        self.n_stocks = n_stocks
        self.n_days = n_days
        self.rng = rng
        self.stock_names = stock_names or [f"股票{i}" for i in range(n_stocks)]
        self.p_policy = p_policy
        self.p_breaking = p_breaking
        self.p_sentiment = p_sentiment
        self.p_earnings = p_earnings

    def generate(self) -> List[NewsEvent]:
        events: List[NewsEvent] = []
        r = self.rng
        for t in range(1, self.n_days):
            if r.random() < self.p_sentiment:
                events.append(self._make_sentiment(t))
            if r.random() < self.p_policy:
                events.append(self._make_policy(t))
            if r.random() < self.p_breaking:
                events.append(self._make_breaking(t))
            if r.random() < self.p_earnings and t % 60 < 5:   # 财报季集中
                events.append(self._make_earnings(t))
        return events

    # ---- 各类型事件生成器 ----

    def _make_sentiment(self, t: int) -> NewsEvent:
        direction, tpl = self.rng.choice(_SENTIMENT_TPL)
        sign = 1.0 if direction == "利好" else -1.0
        mag = sign * self.rng.uniform(0.004, 0.015)
        return NewsEvent(
            day=t, kind="sentiment",
            headline=tpl.format(v=self.rng.integers(50, 500)),
            impacts=np.full(self.n_stocks, mag),
            decay=0.85, duration=10,
        )

    def _make_policy(self, t: int) -> NewsEvent:
        direction, tpl = self.rng.choice(_POLICY_TPL)
        sign = 1.0 if direction == "利好" else -1.0
        k = int(self.rng.integers(0, self.n_stocks))
        mag = sign * self.rng.uniform(0.01, 0.04)
        impacts = np.zeros(self.n_stocks)
        impacts[k] = mag
        # 政策有时会外溢到其他股（弱关联）
        for j in range(self.n_stocks):
            if j != k and self.rng.random() < 0.3:
                impacts[j] = mag * self.rng.uniform(0.2, 0.6)
        return NewsEvent(
            day=t, kind="policy",
            headline=tpl.format(
                industry=self.rng.choice(_INDUSTRIES),
                stock=self.stock_names[k],
            ),
            impacts=impacts,
            decay=0.95, duration=20,
        )

    def _make_breaking(self, t: int) -> NewsEvent:
        direction, tpl = self.rng.choice(_BREAKING_TPL)
        sign = 1.0 if direction == "利好" else -1.0
        k = int(self.rng.integers(0, self.n_stocks))
        mag = sign * self.rng.uniform(0.02, 0.07)
        impacts = np.zeros(self.n_stocks)
        impacts[k] = mag
        return NewsEvent(
            day=t, kind="breaking",
            headline=tpl.format(
                stock=self.stock_names[k],
                v=self.rng.integers(5, 200),
            ),
            impacts=impacts,
            decay=0.6, duration=5,
        )

    def _make_earnings(self, t: int) -> NewsEvent:
        direction, tpl = self.rng.choice(_EARNINGS_TPL)
        sign = 1.0 if direction == "利好" else -1.0
        k = int(self.rng.integers(0, self.n_stocks))
        mag = sign * self.rng.uniform(0.03, 0.10)
        impacts = np.zeros(self.n_stocks)
        impacts[k] = mag
        return NewsEvent(
            day=t, kind="earnings",
            headline=tpl.format(
                stock=self.stock_names[k],
                v=self.rng.integers(10, 80),
            ),
            impacts=impacts,
            decay=0.3, duration=3,
        )


# ============================================================
# 2. 市场模拟器（含事件）
# ============================================================

@dataclass
class MarketConfig:
    n_stocks: int = 3
    n_days: int = 252
    start_price: float = 100.0
    mu: float = 0.10
    bear_mu: float = -0.20
    sigma: float = 0.25
    dt: float = 1.0 / 252
    regime_prob: float = 0.02
    corr: float = 0.5
    seed: Optional[int] = 42
    # 事件概率
    p_policy: float = 0.015
    p_breaking: float = 0.02
    p_sentiment: float = 0.03
    p_earnings: float = 0.01
    stock_names: Optional[List[str]] = None
    # ---- V2: GARCH(1,1) 波动率聚集 ----
    garch_alpha: float = 0.10
    garch_beta: float = 0.85
    # ---- V2: 行业板块联动 ----
    n_industries: int = 2
    industry_corr: float = 0.7
    cross_industry_corr: float = 0.3
    # V3: 三态市场
    bull_sigma: float = 0.20
    range_mu: float = 0.00
    range_sigma: float = 0.25
    bear_sigma: float = 0.40
    regime_switch_bull: float = 0.01
    regime_switch_bear: float = 0.03
    # V3: 黑天鹅
    jump_prob: float = 0.001
    jump_mean: float = -0.15
    jump_std: float = 0.05
    # V3: 停牌/分红
    suspend_prob: float = 0.002
    dividend_prob: float = 0.005
    dividend_yield: float = 0.02


@dataclass
class Market:
    prices: np.ndarray                 # (n_days, n_stocks)
    events: List[NewsEvent]
    stock_names: List[str]


class MarketSimulator:
    def __init__(self, cfg: MarketConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def generate(self) -> Market:
        """V3: 三态 regime + GARCH + 行业 + 黑天鹅 + 停牌 + 分红"""
        c = self.cfg
        n, k = c.n_days, c.n_stocks
        names = c.stock_names or [f"股票{i}" for i in range(k)]

        gen = NewsGenerator(
            n_stocks=k, n_days=n, rng=self.rng, stock_names=names,
            p_policy=c.p_policy, p_breaking=c.p_breaking,
            p_sentiment=c.p_sentiment, p_earnings=c.p_earnings,
        )
        events = gen.generate()

        news_impact = np.zeros((n, k))
        for ev in events:
            for d in range(ev.duration):
                t = ev.day + d
                if t >= n:
                    break
                news_impact[t] += ev.impacts * (ev.decay ** d)

        n_ind = max(1, c.n_industries)
        industry_id = self.rng.integers(0, n_ind, size=k)

        # 三态 regime 参数
        regimes = [
            (c.mu,        c.bull_sigma,  0.30),   # 0: 牛市
            (c.range_mu,  c.range_sigma, 0.50),   # 1: 震荡
            (c.bear_mu,   c.bear_sigma,  0.20),   # 2: 熊市
        ]
        sigma_by_regime = [s for _, s, _ in regimes]

        # GARCH 参数（对每个 regime 分别维护）
        alphas, betas = c.garch_alpha, c.garch_beta
        sigma2_t = c.sigma ** 2
        regime = 1  # 从震荡起步

        # 停牌 / 分红
        suspended = np.zeros((n, k), dtype=bool)
        dividends = np.zeros((n, k))

        prices = np.empty((n, k), dtype=float)
        prices[0] = c.start_price
        sqrt_dt = math.sqrt(c.dt)

        rho_m = float(np.clip(c.cross_industry_corr, 0.0, 0.999))
        rho_ind = max(c.industry_corr - rho_m, 0.0)
        rho_idio = max(1.0 - rho_m - rho_ind, 1e-6)
        s_m, s_i, s_e = math.sqrt(rho_m), math.sqrt(rho_ind), math.sqrt(rho_idio)

        for t in range(1, n):
            # --- regime 切换 ---
            u = self.rng.random()
            if regime == 0:  # 牛市
                if u < c.regime_switch_bear:
                    regime = 2
                elif u < c.regime_switch_bear + 0.02:
                    regime = 1
            elif regime == 2:  # 熊市
                if u < c.regime_switch_bull:
                    regime = 0
                elif u < c.regime_switch_bull + 0.02:
                    regime = 1
            else:  # 震荡
                if u < 0.03:
                    regime = 0
                elif u < 0.06:
                    regime = 2

            mu_now = regimes[regime][0]
            sigma_long = sigma_by_regime[regime]

            # --- GARCH 更新 ---
            sigma_t = math.sqrt(max(sigma2_t, 1e-8))
            z_m = self.rng.normal()
            z_i = self.rng.normal(size=n_ind)
            z_e = self.rng.normal(size=k)
            z = s_m * z_m + s_i * z_i[industry_id] + s_e * z_e

            log_ret = (mu_now - 0.5 * sigma_t ** 2) * c.dt + sigma_t * sqrt_dt * z

            # --- 黑天鹅跳跃 ---
            if self.rng.random() < c.jump_prob:
                jump = self.rng.normal(c.jump_mean, c.jump_std)
                log_ret += jump
                # 单只股票跳跃（不是全部）
                mask = self.rng.random(k) < 0.5
                log_ret = np.where(mask, log_ret, log_ret - jump)  # 恢复未被影响的

            # --- 新闻冲击 ---
            log_ret += news_impact[t]

            # --- 停牌 ---
            if c.suspend_prob > 0:
                susp_today = self.rng.random(k) < c.suspend_prob
                suspended[t] = susp_today
                log_ret = np.where(susp_today, 0.0, log_ret)

            # --- 分红 ---
            if c.dividend_prob > 0:
                div_mask = self.rng.random(k) < c.dividend_prob
                div_amount = c.dividend_yield * prices[t-1] * div_mask
                dividends[t] = div_amount
                # 价格除权
                price_before = prices[t-1]
                price_after = price_before * np.exp(log_ret)
                price_after = np.where(div_mask, price_after - div_amount, price_after)
                prices[t] = np.maximum(price_after, 0.01)
            else:
                prices[t] = prices[t-1] * np.exp(log_ret)

            # GARCH 状态更新
            long_var = sigma_long ** 2
            omega = long_var * (1.0 - alphas - betas)
            sigma2_t = omega + alphas * (sigma_t * z_m) ** 2 + betas * sigma2_t

        return Market(prices=prices, events=events, stock_names=names)


@dataclass
class EnvConfig:
    initial_cash: float = 100_000.0
    fee_rate: float = 0.0003
    slippage: float = 0.0005
    allow_short: bool = False
    window: int = 20
    news_lookback: int = 5             # obs 里回看多少天的新闻
    # V3: A股交易规则
    stamp_tax: float = 0.001          # 印花税（卖出）
    transfer_fee: float = 0.00001     # 过户费
    min_commission: float = 5.0       # 最低佣金
    lot_size: int = 100               # 一手
    price_limit: float = 0.10         # 涨跌停幅度
    t_plus_1: bool = True             # T+1


class TradingEnv:
    def __init__(self, prices: np.ndarray,
                 events: Optional[List[NewsEvent]] = None,
                 stock_names: Optional[List[str]] = None,
                 cfg: Optional[EnvConfig] = None):
        p = np.asarray(prices, dtype=float)
        if p.ndim == 1:
            p = p[:, None]
        self.prices = p
        self.events = events or []
        self.stock_names = stock_names or [f"股票{i}" for i in range(p.shape[1])]
        self.cfg = cfg or EnvConfig()
        self.n_days, self.n_stocks = p.shape
        self.reset()

    def reset(self):
        self.t = 0
        self.cash = float(self.cfg.initial_cash)
        self.shares = np.zeros(self.n_stocks)
        self.equity_curve: List[float] = [self.cash]
        self.trades: List[dict] = []
        self.turnover = 0.0
        self.done = False
        self._bought_today = np.zeros(self.n_stocks, dtype=float)
        return self._obs()

    @property
    def price(self) -> np.ndarray:
        return self.prices[self.t]

    @property
    def equity(self) -> float:
        return self.cash + float(self.shares @ self.price)

    def step(self, action) -> Tuple[dict, float, bool, dict]:
        if self.done:
            raise RuntimeError("episode 已结束，请先 reset()")
        prev = self.equity
        self._rebalance(action)
        self.t += 1
        self.done = self.t >= self.n_days - 1
        new = self.equity
        reward = math.log(new / prev) if prev > 0 and new > 0 else 0.0
        self.equity_curve.append(new)
        return self._obs(), reward, self.done, {"t": self.t, "equity": new}

    def _rebalance(self, action) -> None:
        """V3: 含 T+1、印花税、涨跌停、一手的真实交易"""
        w = np.asarray(action, dtype=float).reshape(-1)
        if w.size != self.n_stocks:
            raise ValueError(f"动作维度应为 {self.n_stocks}，收到 {w.size}")
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 0.0, 1.0) if not self.cfg.allow_short else np.clip(w, -1.0, 1.0)
        if np.abs(w).sum() > 1.0:
            w = w / np.abs(w).sum()

        price = self.price
        prev_price = self.prices[self.t - 1] if self.t > 0 else price

        # 涨跌停判定
        up_limit = prev_price * (1.0 + self.cfg.price_limit)
        down_limit = prev_price * (1.0 - self.cfg.price_limit)
        tradable_buy = price < up_limit * 0.999    # 涨停不能买
        tradable_sell = price > down_limit * 1.001 # 跌停不能卖
        suspended = ~np.isfinite(price) | (price <= 0)

        equity = self.equity
        target_shares = w * equity / np.where(price > 0, price, 1.0)
        # 向下取整为一手
        target_shares = np.floor(target_shares / self.cfg.lot_size) * self.cfg.lot_size

        delta = target_shares - self.shares
        buy = np.maximum(delta, 0.0)
        sell = np.maximum(-delta, 0.0)

        # T+1：今天买入的不能今天卖（用 buy_available 追踪）
        if self.cfg.t_plus_1:
            sell = np.minimum(sell, self.shares - self._bought_today)
        sell = np.maximum(sell, 0.0)

        # 涨跌停/停牌：不可交易的强制为 0
        buy = np.where(tradable_buy & ~suspended, buy, 0.0)
        sell = np.where(tradable_sell & ~suspended, sell, 0.0)

        # 现金约束
        fee_buy = self.cfg.fee_rate + self.cfg.transfer_fee
        buy_cost_raw = (buy * price * (1 + fee_buy)).sum()
        if buy_cost_raw > self.cash and buy_cost_raw > 0:
            scale = self.cash / buy_cost_raw
            buy *= scale
            buy = np.floor(buy / self.cfg.lot_size) * self.cfg.lot_size

        # 计算成本
        buy_value = float((buy * price).sum())
        sell_value = float((sell * price).sum())
        commission_buy = max(buy_value * self.cfg.fee_rate, self.cfg.min_commission if buy_value > 0 else 0)
        commission_sell = max(sell_value * self.cfg.fee_rate, self.cfg.min_commission if sell_value > 0 else 0)
        stamp_tax = sell_value * self.cfg.stamp_tax
        transfer_fee = (buy_value + sell_value) * self.cfg.transfer_fee
        total_cost = commission_buy + commission_sell + stamp_tax + transfer_fee

        # 交易
        self.shares = self.shares + buy - sell
        self.cash = self.cash - buy_value - commission_buy - transfer_fee / 2 + sell_value - commission_sell - stamp_tax - transfer_fee / 2

        self._bought_today = buy.copy()   # 标记今日买入（明日才可卖）
        traded_value = buy_value + sell_value
        self.turnover += traded_value

        if traded_value > 1e-6:
            self.trades.append({
                "t": self.t,
                "buy": buy.tolist(),
                "sell": sell.tolist(),
                "value": float(buy_value - sell_value),
                "cost": float(total_cost),
            })

    def _recent_news(self) -> List[dict]:
        lo = self.t - self.cfg.news_lookback
        out = []
        for ev in self.events:
            if lo < ev.day <= self.t:
                out.append({
                    "day": ev.day,
                    "kind": ev.kind,
                    "headline": ev.headline,
                })
        return out

    def _obs(self) -> dict:
        lo = max(0, self.t - self.cfg.window + 1)
        equity = self.equity
        pos_val = float((self.shares * self.price).sum())
        return {
            "t": self.t,
            "n_days": self.n_days,
            "n_stocks": self.n_stocks,
            "stock_names": self.stock_names,
            "prices": self.prices[lo:self.t + 1].copy(),
            "last_prices": self.price.copy(),
            "cash": self.cash,
            "shares": self.shares.copy(),
            "equity": equity,
            "position_ratio": pos_val / equity if equity > 0 else 0.0,
            "news": self._recent_news(),
        }


# ============================================================
# 4. 指标（不变）
# ============================================================

def compute_metrics(eq_curve, periods_per_year: int = 252) -> Dict[str, float]:
    eq = np.asarray(eq_curve, dtype=float)
    eq = eq[eq > 0]
    if eq.size < 2:
        return {k: 0.0 for k in ("total_return", "annual_return", "volatility",
                                 "sharpe", "max_drawdown", "calmar", "win_rate")}
    rets = np.diff(eq) / eq[:-1]
    n = rets.size
    total = float(eq[-1] / eq[0] - 1.0)
    ann = float((1 + total) ** (periods_per_year / max(n, 1)) - 1.0)
    std = float(rets.std(ddof=1)) if n > 1 else 0.0
    vol = std * math.sqrt(periods_per_year)
    sharpe = float(rets.mean() / std * math.sqrt(periods_per_year)) if std > 1e-12 else 0.0
    peak = np.maximum.accumulate(eq)
    mdd = float(((eq - peak) / peak).min())
    calmar = float(ann / abs(mdd)) if mdd < -1e-9 else 0.0
    return {"total_return": total, "annual_return": ann, "volatility": vol,
            "sharpe": sharpe, "max_drawdown": mdd, "calmar": calmar,
            "win_rate": float((rets > 0).mean()) if n else 0.0}


# ============================================================
# 5. Agent 基类 + 基线
# ============================================================

class Agent(abc.ABC):
    def reset(self) -> None: ...
    @abc.abstractmethod
    def act(self, obs: dict) -> np.ndarray: ...


class BuyAndHoldAgent(Agent):
    def __init__(self, n: int = 1): self.n = n; self._done = False
    def reset(self): self._done = False
    def act(self, obs):
        if self._done: return np.zeros(self.n)
        self._done = True
        return np.ones(self.n) / self.n


class RandomAgent(Agent):
    def __init__(self, n: int = 1, seed: Optional[int] = None):
        self.n = n; self.rng = np.random.default_rng(seed)
    def act(self, obs): return self.rng.random(self.n)


class MomentumAgent(Agent):
    def __init__(self, n: int = 1, window: int = 20):
        self.n = n; self.window = window
    def act(self, obs):
        p = obs["prices"]
        if p.shape[0] < 2: return np.zeros(self.n)
        h = p[-min(self.window, p.shape[0]):]
        sig = (h[-1] / h[0] - 1 > 0).astype(float)
        return sig / sig.sum() if sig.sum() > 0 else np.zeros(self.n)


# ============================================================
# 6. 通用 AI Agent（OpenAI 兼容端点）
# ============================================================

@dataclass
class ModelEndpoint:
    """一个参赛模型的配置"""
    name: str
    base_url: str
    model_id: str
    api_key_env: str                     # 环境变量名，如 "DEEPSEEK_API_KEY"
    extra: Dict[str, Any] = field(default_factory=dict)  # 透传给 API 的额外参数


# ★★★ 请在这里填入各模型的实际调用信息 ★★★
MODELS: Dict[str, ModelEndpoint] = {
    # ---- DeepSeek 官方 ----
    "deepseek-v4.1-flash": ModelEndpoint(
        name="deepseek-v4.1-flash",
        base_url="https://api.deepseek.com",
        model_id="deepseek-flash",
        api_key_env="DEEPSEEK_API_KEY",
        extra={"reasoning_effort": "low"},
    ),
    "deepseek-v4-pro": ModelEndpoint(
        name="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        model_id="deepseek-v4-pro",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    # ---- ECNU ----
    "ecnu-plus": ModelEndpoint(
        name="ecnu-plus",
        base_url="https://chat.ecnu.edu.cn/open/api/v1",
        model_id="ecnu-plus",
        api_key_env="ECNU_API_KEY",
    ),
    "ecnu-max": ModelEndpoint(
        name="ecnu-max",
        base_url="https://chat.ecnu.edu.cn/open/api/v1",
        model_id="ecnu-max",
        api_key_env="ECNU_API_KEY",
    ),
    # ---- 智谱 BigModel ----
    "GLM-4.7-Flash": ModelEndpoint(
        name="GLM-4.7-Flash",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model_id="glm-4.7-flash",
        api_key_env="ZHIPU_API_KEY",
    ),
    # ---- SiliconFlow 免费模型 ----
    "Xing4.0-29B": ModelEndpoint(
        name="Xing4.0-29B",
        base_url="https://api.siliconflow.cn/v1",
        model_id="XingChenAGI/Xing4.0-29B",
        api_key_env="SILICONFLOW_API_KEY",
    ),
    "Qwen3-8B": ModelEndpoint(
        name="Qwen3-8B",
        base_url="https://api.siliconflow.cn/v1",
        model_id="Qwen/Qwen3-8B",
        api_key_env="SILICONFLOW_API_KEY",
    ),
    # ---- SiliconFlow 免费模型：GLM 系列 ----
    "GLM-Z1-9B": ModelEndpoint(
        name="GLM-Z1-9B",
        base_url="https://api.siliconflow.cn/v1",
        model_id="THUDM/GLM-Z1-9B-0414",
        api_key_env="SILICONFLOW_API_KEY",
    ),
    "GLM-4-9B": ModelEndpoint(
        name="GLM-4-9B",
        base_url="https://api.siliconflow.cn/v1",
        model_id="THUDM/GLM-4-9B-0414",
        api_key_env="SILICONFLOW_API_KEY",
    ),
    "DeepSeek-R1-8B": ModelEndpoint(
        name="DeepSeek-R1-8B",
        base_url="https://api.siliconflow.cn/v1",
        model_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        api_key_env="SILICONFLOW_API_KEY",
    ),
    "glm-5.3-flash": ModelEndpoint(
        name="glm-5.3-flash",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model_id="glm-5.3-flash",
        api_key_env="ZHIPU_API_KEY",
    ),
    "glm-5.3-flashx": ModelEndpoint(
        name="glm-5.3-flashx",
        base_url="https://open.bigmodel.cn/api/paas/v4",
        model_id="glm-5.3-flashx",
        api_key_env="ZHIPU_API_KEY",
    ),
    # ---- CodeCraft API ----
    "CC-Claude-Sonnet-5": ModelEndpoint(
        name="CC-Claude-Sonnet-5",
        base_url="https://codecraftapi.com/v1",
        model_id="claude-sonnet-5",
        api_key_env="CODECRAFT_API_KEY",
    ),
    "CC-GPT-5.6-Luna": ModelEndpoint(
        name="CC-GPT-5.6-Luna",
        base_url="https://codecraftapi.com/v1",
        model_id="gpt-5.6-luna",
        api_key_env="CODECRAFT_API_KEY",
    ),
    # ---- SiliconFlow（聚合平台，模型名待确认）----
}


_SYSTEM_PROMPT = """你是一个量化交易 AI。你将看到历史行情、账户状态、以及近期新闻/政策/市场情绪。
请决定每只股票的目标仓位权重（占当前总资产的比例）。

规则：
- 权重范围 [0, 1]（不做空），所有股票权重之和 <= 1
- 不交易时输出全 0 即可
- 请结合新闻判断事件的方向和影响范围

只输出 JSON，无任何解释或 markdown：
{"weights": [0.5, 0.3, 0.2], "reason": "简短理由"}
"""


class RemoteAgent(Agent):
    """
    通用远程/本地 AI Agent，走 OpenAI 兼容 Chat Completions。
    适用于 DeepSeek / GLM / vLLM / SGLang / Ollama / LM Studio 等。
    """

    def __init__(self, endpoint: ModelEndpoint,
                 n_stocks: int, stock_names: Optional[List[str]] = None,
                 window: int = 20, temperature: float = 0.0,
                 max_tokens: int = 16384, timeout: float = 60.0,
                 fallback: Optional[np.ndarray] = None,
                 verbose: bool = False):
        self.endpoint = endpoint
        self.n_stocks = n_stocks
        self.stock_names = stock_names or [f"股票{i}" for i in range(n_stocks)]
        self.window = window
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.fallback = fallback if fallback is not None else np.zeros(n_stocks)
        self.verbose = verbose
        self.n_calls = 0
        self.n_parse_failures = 0
        self._client = None
        import os as _os
        _ld = _os.environ.get("STOCK_SIM_LOG_DIR")
        if _ld:
            _os.makedirs(_ld, exist_ok=True)
            self._decision_log_path = _os.path.join(_ld, f"{endpoint.name}.jsonl")
        else:
            self._decision_log_path = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError("请先 pip install openai") from e
            api_key = os.environ.get(self.endpoint.api_key_env, "EMPTY")
            self._client = OpenAI(
                base_url=self.endpoint.base_url or None,
                api_key=api_key,
                timeout=self.timeout,
            )
        return self._client

    def build_prompt(self, obs: dict) -> str:
        lines = [_SYSTEM_PROMPT, ""]
        lines.append("=== 股票列表 ===")
        for i, nm in enumerate(self.stock_names):
            lines.append(f"  [{i}] {nm}")

        lines.append("\n=== 近期新闻/政策/情绪 ===")
        news = obs.get("news", [])
        if not news:
            lines.append("  （无）")
        for n in news:
            lines.append(f"  第{n['day']}日 [{n['kind']}] {n['headline']}")

        p = obs["prices"]
        lines.append("\n=== 最近收盘价（越右越新）===")
        for i, nm in enumerate(self.stock_names):
            series = ", ".join(f"{x:.2f}" for x in p[:, i])
            lines.append(f"  {nm}: {series}")

        lines += [
            "\n=== 账户状态 ===",
            f"  交易日: {obs['t']} / {obs['n_days']}",
            f"  现金: {obs['cash']:.2f}",
            f"  持仓: {np.array2string(obs['shares'], precision=2)}",
            f"  最新价: {np.array2string(obs['last_prices'], precision=2)}",
            f"  总资产: {obs['equity']:.2f}",
            f"  当前仓位: {obs['position_ratio']*100:.1f}%",
            "",
            f"请给出 {self.n_stocks} 个权重，输出 JSON：",
        ]
        return "\n".join(lines)

    def parse(self, text: str) -> Optional[np.ndarray]:
        if not text:
            return None
        try:
            s, e = text.index("{"), text.rindex("}")
            data = json.loads(text[s:e + 1])
            w = data.get("weights") or data.get("w")
            if w is not None:
                arr = np.asarray(w, dtype=float).reshape(-1)
                if arr.size == self.n_stocks:
                    return arr
                if arr.size == 1:
                    return np.repeat(arr, self.n_stocks)
        except Exception:
            pass
        m = re.search(r"\[([^\]]*)\]", text)
        if m:
            nums = re.findall(r"-?\d+\.?\d*", m.group(1))
            if len(nums) >= self.n_stocks:
                return np.array([float(x) for x in nums[:self.n_stocks]])
        return None

    def act(self, obs: dict) -> np.ndarray:
        prompt = self.build_prompt(obs)
        raw = ""
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=self.endpoint.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                **self.endpoint.extra,
            )
            raw = resp.choices[0].message.content or ""
            self.n_calls += 1
        except Exception as e:
            self.n_parse_failures += 1
            if self.verbose:
                print(f"[RemoteAgent:{self.endpoint.name}] 调用失败: {e}")

        action = self.parse(raw)
        reason = ""
        if action is None:
            self.n_parse_failures += 1
            action = self.fallback.copy()
        else:
            try:
                _s, _e = raw.index("{"), raw.rindex("}")
                _d = json.loads(raw[_s:_e+1])
                reason = str(_d.get("reason", ""))
            except Exception:
                pass
        if self._decision_log_path:
            try:
                with open(self._decision_log_path, "a", encoding="utf-8") as _f:
                    _f.write(json.dumps({
                        "model": self.endpoint.name,
                        "t": int(obs["t"]),
                        "equity": float(obs["equity"]),
                        "position_ratio": float(obs["position_ratio"]),
                        "news": [n["headline"] for n in obs.get("news", [])],
                        "weights": action.tolist(),
                        "reason": reason,
                    }, ensure_ascii=False) + "\n")
            except Exception:
                pass
        if self.verbose:
            print(f"\n===== {self.endpoint.name} @ t={obs['t']} =====")
            print(raw)
        return action


# ============================================================
# 7. 回测 & 批量评测
# ============================================================

@dataclass
class BacktestResult:
    equity_curve: np.ndarray
    metrics: Dict[str, float]
    env: TradingEnv


def run_backtest(agent: Agent, market: Market,
                 env_cfg: Optional[EnvConfig] = None,
                 step_delay: float = 0.0,
                 live: bool = False) -> BacktestResult:
    import time
    env = TradingEnv(market.prices, market.events, market.stock_names, env_cfg)
    agent.reset()
    obs = env.reset()
    done = False
    while not done:
        if live:
            t = obs["t"]
            news = obs.get("news", [])
            news_str = " | ".join(n["headline"] for n in news) if news else "-"
            print(f"[Day {t:3d}/{obs['n_days']-1}] "
                  f"资产={obs['equity']:>12,.2f}  "
                  f"仓位={obs['position_ratio']*100:5.1f}%  "
                  f"新闻: {news_str[:50]}")
        a = agent.act(obs)
        if live:
            print(f"             → AI 决策: {np.array2string(np.asarray(a), precision=3)}")
        obs, _r, done, _i = env.step(a)
        if step_delay > 0:
            time.sleep(step_delay)
    eq = np.asarray(env.equity_curve, dtype=float)
    m = compute_metrics(eq)
    m["n_trades"] = float(len(env.trades))
    m["turnover"] = float(env.turnover)
    m["parse_failures"] = float(getattr(agent, "n_parse_failures", 0))
    m["api_calls"] = float(getattr(agent, "n_calls", 0))
    return BacktestResult(equity_curve=eq, metrics=m, env=env)


def benchmark(agent_factory: Callable[[], Agent],
              market_cfg: Optional[MarketConfig] = None,
              env_cfg: Optional[EnvConfig] = None,
              n_episodes: int = 20,
              base_seed: int = 1000,
              step_delay: float = 0.0,
              live: bool = False) -> Dict[str, Dict[str, float]]:
    market_cfg = market_cfg or MarketConfig()
    env_cfg = env_cfg or EnvConfig()
    rows: List[Dict[str, float]] = []
    for i in range(n_episodes):
        cfg = replace(market_cfg, seed=base_seed + i)
        market = MarketSimulator(cfg).generate()
        agent = agent_factory()
        res = run_backtest(agent, market, env_cfg, step_delay=step_delay, live=live)
        rows.append(res.metrics)
    keys = ["total_return", "annual_return", "volatility", "sharpe",
            "max_drawdown", "calmar", "win_rate", "n_trades",
            "parse_failures", "api_calls"]
    return {
        k: {
            "mean": float(np.mean([r.get(k, 0.0) for r in rows])),
            "std":  float(np.std([r.get(k, 0.0) for r in rows], ddof=1))
                    if len(rows) > 1 else 0.0,
        } for k in keys
    }


# ============================================================
# 8. 命令行入口
# ============================================================

def _print_results(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    hdr = f"{'策略/模型':<24}{'总收益':<24}{'夏普':<10}{'最大回撤':<12}{'胜率':<10}{'调用':<8}"
    print(hdr); print("-" * len(hdr))
    for name, agg in results.items():
        tr = agg["total_return"]
        print(
            f"{name:<24}"
            f"{tr['mean']:+.4f} ± {tr['std']:.4f}    "
            f"{agg['sharpe']['mean']:+.2f}     "
            f"{agg['max_drawdown']['mean']:+.4f}     "
            f"{agg['win_rate']['mean']:.3f}     "
            f"{agg['api_calls']['mean']:.0f}"
        )


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--stocks", type=int, default=3)
    ap.add_argument("--days", type=int, default=252)
    ap.add_argument("--cash", type=float, default=100_000.0)
    ap.add_argument("--allow-short", action="store_true")
    ap.add_argument("--models", type=str, default="",
                    help="逗号分隔的模型名（来自 MODELS），为空则只跑基线")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--step-delay", type=float, default=0.0)
    ap.add_argument("--live", action="store_true")
    args = ap.parse_args()

    names = [f"股票{i}" for i in range(args.stocks)]
    market_cfg = MarketConfig(n_stocks=args.stocks, n_days=args.days,
                              stock_names=names)
    env_cfg = EnvConfig(initial_cash=args.cash, allow_short=args.allow_short)

    factories: Dict[str, Callable[[], Agent]] = {
        "BuyAndHold": lambda: BuyAndHoldAgent(args.stocks),
        "Random":     lambda: RandomAgent(args.stocks, seed=0),
        "Momentum20": lambda: MomentumAgent(args.stocks, window=20),
    }

    if args.models:
        for name in [x.strip() for x in args.models.split(",") if x.strip()]:
            if name not in MODELS:
                print(f"[警告] 未知模型 {name}，跳过")
                continue
            ep = MODELS[name]
            factories[name] = (lambda ep=ep: RemoteAgent(
                endpoint=ep, n_stocks=args.stocks, stock_names=names,
                verbose=args.verbose,
            ))

    print(f"评测: {args.episodes} 轮 × {args.days} 天 × {args.stocks} 股\n")
    results = {}
    for name, fac in factories.items():
        print(f"  ▶ {name} ...")
        results[name] = benchmark(fac, market_cfg, env_cfg, args.episodes,
                                  step_delay=args.step_delay, live=args.live)
    print()
    _print_results(results)


if __name__ == "__main__":
    main()