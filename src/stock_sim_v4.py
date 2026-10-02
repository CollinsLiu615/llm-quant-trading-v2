# -*- coding: utf-8 -*-
"""
stock_sim_v4.py — A 股逼真版市场模拟器（第二阶段）

与 v3 的区别：
  1. 默认 750 只股票，覆盖 4 个板块
  2. 板块化涨跌停：主板 ±10% / 创业板·科创板 ±20% / 北交所 ±30% / ST ±5%
  3. 10×10 行业相关矩阵（替代 v3 的 "1 行业相关度"）
  4. 印花税万分之五（2023-08 减半后的真实税率）
  5. 股票命名带板块/行业前缀

与 v3 的兼容：
  - 接口签名完全兼容（MarketConfig, MarketSimulator, Market,
    EnvConfig, TradingEnv, Agent, run_backtest, compute_metrics）
  - 旧代码只需把 import 从 stock_sim_v3 改为 stock_sim_v4
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
# 0. 板块 & 行业定义
# ============================================================

# 板块 → 涨跌停幅度
SECTOR_LIMITS = {
    "主板":   0.10,
    "创业板": 0.20,
    "科创板": 0.20,
    "北交所": 0.30,
}

# 板块在 750 只股票中的占比
SECTOR_RATIOS = {
    "主板":   0.59,
    "创业板": 0.25,
    "科创板": 0.11,
    "北交所": 0.05,
}

# 10 个真实行业
INDUSTRY_NAMES = [
    "新能源", "半导体", "医药", "白酒", "地产",
    "银行", "军工", "AI算力", "消费", "券商",
]

# 10×10 行业相关矩阵（对称）
INDUSTRY_CORR = np.array([
    #新能源 半导体 医药  白酒  地产  银行  军工  AI   消费  券商
    [1.00, 0.65, 0.30, 0.25, 0.35, 0.20, 0.40, 0.55, 0.35, 0.25],
    [0.65, 1.00, 0.30, 0.20, 0.25, 0.15, 0.45, 0.75, 0.30, 0.30],
    [0.30, 0.30, 1.00, 0.35, 0.25, 0.25, 0.25, 0.30, 0.45, 0.20],
    [0.25, 0.20, 0.35, 1.00, 0.30, 0.35, 0.20, 0.20, 0.60, 0.30],
    [0.35, 0.25, 0.25, 0.30, 1.00, 0.60, 0.20, 0.20, 0.35, 0.40],
    [0.20, 0.15, 0.25, 0.35, 0.60, 1.00, 0.20, 0.15, 0.35, 0.80],
    [0.40, 0.45, 0.25, 0.20, 0.20, 0.20, 1.00, 0.35, 0.25, 0.25],
    [0.55, 0.75, 0.30, 0.20, 0.20, 0.15, 0.35, 1.00, 0.30, 0.25],
    [0.35, 0.30, 0.45, 0.60, 0.35, 0.35, 0.25, 0.30, 1.00, 0.35],
    [0.25, 0.30, 0.20, 0.30, 0.40, 0.80, 0.25, 0.25, 0.35, 1.00],
])


# ============================================================
# 1. 新闻事件（v4：从新闻库抽样，不用硬编码模板）
# ============================================================

@dataclass
class NewsEvent:
    day: int
    kind: str
    headline: str
    impacts: np.ndarray
    decay: float
    duration: int
    industry: str = ""
    direction: str = ""
    source: str = ""


class NewsLibrary:
    """加载并索引 data/news_library.json，按 (kind, industry, direction) 索引。"""

    def __init__(self, path: Optional[str] = None):
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "news_library.json")
        self.path = path
        self._index: Dict[Tuple[str, str, str], List[dict]] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"新闻库不存在: {self.path}")
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            key = (item["kind"], item["industry"], item["direction"])
            self._index.setdefault(key, []).append(item)
        self._loaded = True

    def sample(self, kind: str, industry: str, direction: str,
               rng: np.random.Generator) -> Optional[dict]:
        self.load()
        pool = self._index.get((kind, industry, direction), [])
        if not pool:
            return None
        return pool[int(rng.integers(0, len(pool)))]

    def stats(self) -> Dict[str, int]:
        self.load()
        return {f"{k[0]}|{k[1]}|{k[2]}": len(v) for k, v in self._index.items()}


class NewsGenerator:
    """v4: 从新闻库按 seed 抽样，保证同 seed 同序列可复现。

    触发概率比 v3 高，让 252 天产生 ~100~150 条新闻。
    """

    def __init__(self, n_stocks: int, n_days: int, rng: np.random.Generator,
                 stock_names: Optional[List[str]] = None,
                 industry_of_stock: Optional[np.ndarray] = None,
                 library: Optional[NewsLibrary] = None,
                 p_policy: float = 0.03,
                 p_breaking: float = 0.05,
                 p_sentiment: float = 0.06,
                 p_earnings: float = 0.02):
        self.n_stocks = n_stocks
        self.n_days = n_days
        self.rng = rng
        self.stock_names = stock_names or [f"股票{i}" for i in range(n_stocks)]
        self.industry_of_stock = (
            np.asarray(industry_of_stock)
            if industry_of_stock is not None
            else rng.integers(0, len(INDUSTRY_NAMES), size=n_stocks)
        )
        self.library = library or NewsLibrary()
        self.p_policy = p_policy
        self.p_breaking = p_breaking
        self.p_sentiment = p_sentiment
        self.p_earnings = p_earnings
        self.n_industries = len(INDUSTRY_NAMES)

    def generate(self) -> List[NewsEvent]:
        events: List[NewsEvent] = []
        r = self.rng
        for t in range(1, self.n_days):
            for kind, prob in (
                ("policy",    self.p_policy),
                ("breaking",  self.p_breaking),
                ("sentiment", self.p_sentiment),
                ("earnings",  self.p_earnings),
            ):
                if r.random() < prob:
                    ev = self._make_event(t, kind)
                    if ev is not None:
                        events.append(ev)
        return events

    def _make_event(self, t: int, kind: str) -> Optional[NewsEvent]:
        ind_idx = int(self.rng.integers(0, self.n_industries))
        industry = INDUSTRY_NAMES[ind_idx]
        direction = "利好" if self.rng.random() < 0.5 else "利空"

        item = self.library.sample(kind, industry, direction, self.rng)
        if item is None:
            return None

        mag = float(item["magnitude"])
        sign = 1.0 if direction == "利好" else -1.0

        impacts = np.zeros(self.n_stocks)
        mask = (self.industry_of_stock == ind_idx)
        n_in = int(mask.sum())
        if n_in > 0:
            noise = self.rng.normal(0.0, 0.2, size=n_in)
            impacts[mask] = sign * mag * (1.0 + noise)

        # sentiment 类溢出到全市场（弱影响）
        if kind == "sentiment" and n_in < self.n_stocks:
            other = ~mask
            impacts[other] = sign * mag * 0.15 * self.rng.normal(0, 1, size=int(other.sum()))

        return NewsEvent(
            day=t, kind=kind, headline=item["headline"],
            impacts=impacts,
            decay=float(item["decay"]),
            duration=int(item["duration"]),
            industry=industry,
            direction=direction,
            source=item.get("source", ""),
        )


# ============================================================
# 2. 市场模拟器（v4：板块 + ST + 行业矩阵）
# ============================================================

@dataclass
class MarketConfig:
    n_stocks: int = 750
    n_days: int = 252
    start_price: float = 100.0
    mu: float = 0.10
    bear_mu: float = -0.20
    sigma: float = 0.25
    dt: float = 1.0 / 252
    seed: Optional[int] = 42

    # 板块
    sector_ratios: Optional[Dict[str, float]] = None
    st_ratio: float = 0.03
    st_limit: float = 0.05

    # 新闻（偏多档）
    p_policy: float = 0.12
    p_breaking: float = 0.18
    p_sentiment: float = 0.25
    p_earnings: float = 0.08

    # GARCH
    garch_alpha: float = 0.10
    garch_beta: float = 0.85

    # 行业
    n_industries: int = 10
    cross_industry_corr: float = 0.35
    within_industry_corr: float = 0.55

    # 三态市场
    bull_sigma: float = 0.20
    range_mu: float = 0.00
    range_sigma: float = 0.25
    bear_sigma: float = 0.40
    regime_switch_bull: float = 0.01
    regime_switch_bear: float = 0.03

    # 黑天鹅
    jump_prob: float = 0.001
    jump_mean: float = -0.15
    jump_std: float = 0.05

    # 停牌 / 分红
    suspend_prob: float = 0.002
    dividend_prob: float = 0.005
    dividend_yield: float = 0.02

    stock_names: Optional[List[str]] = None


@dataclass
class Market:
    prices: np.ndarray
    events: List[NewsEvent]
    stock_names: List[str]
    industry_id: np.ndarray
    sectors: List[str]
    price_limits: np.ndarray
    st_mask: np.ndarray


class MarketSimulator:
    def __init__(self, cfg: MarketConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _assign_sectors(self, k):
        ratios = self.cfg.sector_ratios or SECTOR_RATIOS
        total = sum(ratios.values())
        counts = {s: int(k * r / total) for s, r in ratios.items()}
        assigned = sum(counts.values())
        counts["主板"] = counts.get("主板", 0) + (k - assigned)
        sectors = []
        for s, c in counts.items():
            sectors.extend([s] * c)
        self.rng.shuffle(sectors)
        return sectors

    def _assign_industries(self, k):
        """按 INDUSTRY_WEIGHTS 分配行业。"""
        weights = np.array([INDUSTRY_WEIGHTS.get(n, 1.0 / len(INDUSTRY_NAMES))
                            for n in INDUSTRY_NAMES])
        weights = weights / weights.sum()
        counts = np.floor(weights * k).astype(int)
        shortfall = k - counts.sum()
        if shortfall > 0:
            order = np.argsort(-weights)
            for i in range(shortfall):
                counts[order[i % len(order)]] += 1
        industry_id = np.repeat(np.arange(len(INDUSTRY_NAMES)), counts)
        self.rng.shuffle(industry_id)
        return industry_id

    def _assign_st(self, k):
        return self.rng.random(k) < self.cfg.st_ratio

    def _make_names(self, k, sectors, industry_id, st_mask):
        abbr = {"主板": "SH", "创业板": "SZ", "科创板": "KC", "北交所": "BJ"}
        names = []
        for i in range(k):
            prefix = "ST" if st_mask[i] else ""
            names.append(
                f"{prefix}{abbr[sectors[i]]}{i:04d}_{INDUSTRY_NAMES[industry_id[i]]}"
            )
        return names

    def _price_limits(self, sectors, st_mask):
        limits = np.array([SECTOR_LIMITS[s] for s in sectors])
        limits[st_mask] = self.cfg.st_limit
        return limits

    def generate(self) -> Market:
        c = self.cfg
        n, k = c.n_days, c.n_stocks

        sectors = self._assign_sectors(k)
        industry_id = self._assign_industries(k)
        st_mask = self._assign_st(k)
        price_limits = self._price_limits(sectors, st_mask)
        names = c.stock_names or self._make_names(k, sectors, industry_id, st_mask)

        gen = NewsGenerator(
            n_stocks=k, n_days=n, rng=self.rng, stock_names=names,
            industry_of_stock=industry_id,
            p_policy=c.p_policy, p_breaking=c.p_breaking,
            p_sentiment=c.p_sentiment, p_earnings=c.p_earnings,
        )
        events = gen.generate()

        # v4 修正：新闻冲击改为"当天跳空 + 之后反转"，
        # 避免"持续 N 天慢涨"造成可预测套利。
        news_impact = np.zeros((n, k))
        for ev in events:
            # 当天：全量冲击
            if ev.day < n:
                news_impact[ev.day] += ev.impacts
            # 之后：每天反向修正，10 天内回吐 60% 冲击（真实市场常见）
            reversal_days = 10
            reversal_total = 0.6
            per_day = ev.impacts * reversal_total / reversal_days
            for d in range(1, reversal_days + 1):
                t = ev.day + d
                if t >= n:
                    break
                news_impact[t] -= per_day

        # 行业相关矩阵 → Cholesky
        corr = INDUSTRY_CORR.copy()
        corr[np.diag_indices_from(corr)] += 1e-6
        L = np.linalg.cholesky(corr)

        regimes = [
            (c.mu,       c.bull_sigma),
            (c.range_mu, c.range_sigma),
            (c.bear_mu,  c.bear_sigma),
        ]
        alphas, betas = c.garch_alpha, c.garch_beta
        sigma2_t = c.sigma ** 2
        regime = 1

        suspended = np.zeros((n, k), dtype=bool)
        dividends = np.zeros((n, k))

        prices = np.empty((n, k), dtype=float)
        prices[0] = c.start_price
        sqrt_dt = math.sqrt(c.dt)

        rho_m = c.cross_industry_corr
        rho_ind = max(c.within_industry_corr - rho_m, 0.0)
        rho_idio = max(1.0 - rho_m - rho_ind, 1e-6)
        s_m = math.sqrt(rho_m)
        s_ind = math.sqrt(rho_ind)
        s_e = math.sqrt(rho_idio)

        n_ind = len(INDUSTRY_NAMES)

        for t in range(1, n):
            u = self.rng.random()
            if regime == 0:
                if u < c.regime_switch_bear:
                    regime = 2
                elif u < c.regime_switch_bear + 0.02:
                    regime = 1
            elif regime == 2:
                if u < c.regime_switch_bull:
                    regime = 0
                elif u < c.regime_switch_bull + 0.02:
                    regime = 1
            else:
                if u < 0.03:
                    regime = 0
                elif u < 0.06:
                    regime = 2

            mu_now = regimes[regime][0]
            sigma_long = regimes[regime][1]

            sigma_t = math.sqrt(max(sigma2_t, 1e-8))
            z_m = self.rng.normal()
            eps = self.rng.normal(size=n_ind)
            z_ind_vec = L @ eps
            z_industry = z_ind_vec[industry_id]
            z_idio = self.rng.normal(size=k)

            z = s_m * z_m + s_ind * z_industry + s_e * z_idio
            log_ret = (mu_now - 0.5 * sigma_t ** 2) * c.dt + sigma_t * sqrt_dt * z

            if self.rng.random() < c.jump_prob:
                jump = self.rng.normal(c.jump_mean, c.jump_std)
                mask = self.rng.random(k) < 0.5
                log_ret = np.where(mask, log_ret + jump, log_ret)

            log_ret += news_impact[t]

            if c.suspend_prob > 0:
                susp_today = self.rng.random(k) < c.suspend_prob
                suspended[t] = susp_today
                log_ret = np.where(susp_today, 0.0, log_ret)

            if c.dividend_prob > 0:
                div_mask = self.rng.random(k) < c.dividend_prob
                div_amount = c.dividend_yield * prices[t-1] * div_mask
                dividends[t] = div_amount
                price_after = prices[t-1] * np.exp(log_ret)
                price_after = np.where(div_mask, price_after - div_amount, price_after)
                prices[t] = np.maximum(price_after, 0.01)
            else:
                prices[t] = prices[t-1] * np.exp(log_ret)

            long_var = sigma_long ** 2
            omega = long_var * (1.0 - alphas - betas)
            sigma2_t = omega + alphas * (sigma_t * z_m) ** 2 + betas * sigma2_t

        return Market(
            prices=prices,
            events=events,
            stock_names=names,
            industry_id=industry_id,
            sectors=sectors,
            price_limits=price_limits,
            st_mask=st_mask,
        )


# ============================================================
# 3. 交易环境（v4：板块涨跌停 + ST 5%）
# ============================================================

@dataclass
class EnvConfig:
    initial_cash: float = 1_000_000.0
    fee_rate: float = 0.0003
    slippage: float = 0.0005
    allow_short: bool = False
    window: int = 60
    news_lookback: int = 5
    stamp_tax: float = 0.0005        # v4: 万分之五（2023-08 后）
    transfer_fee: float = 0.00001
    min_commission: float = 5.0
    lot_size: int = 100
    t_plus_1: bool = True


class TradingEnv:
    def __init__(self, market: Market,
                 cfg: Optional[EnvConfig] = None):
        self.market = market
        self.prices = np.asarray(market.prices, dtype=float)
        self.price_limits = np.asarray(market.price_limits, dtype=float)
        self.stock_names = market.stock_names
        self.events = market.events
        self.cfg = cfg or EnvConfig()
        self.n_days, self.n_stocks = self.prices.shape
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
        w = np.asarray(action, dtype=float).reshape(-1)
        if w.size != self.n_stocks:
            raise ValueError(f"动作维度应为 {self.n_stocks}，收到 {w.size}")
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 0.0, 1.0) if not self.cfg.allow_short else np.clip(w, -1.0, 1.0)
        if w.sum() > 1.0:
            w = w / w.sum()

        price_mid = self.price
        # 滑点：买单价格上浮，卖单价格下调
        slip = self.cfg.slippage
        price = price_mid.copy()
        prev_price = self.prices[self.t - 1] if self.t > 0 else price_mid

        # v4: 每只股票用自己的涨跌停幅度
        up_limit = prev_price * (1.0 + self.price_limits)
        down_limit = prev_price * (1.0 - self.price_limits)
        tradable_buy = price < up_limit * 0.999
        tradable_sell = price > down_limit * 1.001
        suspended = ~np.isfinite(price) | (price <= 0)

        equity = self.equity
        target_shares = w * equity / np.where(price > 0, price, 1.0)
        target_shares = np.floor(target_shares / self.cfg.lot_size) * self.cfg.lot_size

        delta = target_shares - self.shares
        buy = np.maximum(delta, 0.0)
        sell = np.maximum(-delta, 0.0)

        if self.cfg.t_plus_1:
            sell = np.minimum(sell, self.shares - self._bought_today)
        sell = np.maximum(sell, 0.0)

        buy = np.where(tradable_buy & ~suspended, buy, 0.0)
        sell = np.where(tradable_sell & ~suspended, sell, 0.0)

        fee_buy = self.cfg.fee_rate + self.cfg.transfer_fee
        buy_cost_raw = (buy * price * (1 + fee_buy)).sum()
        if buy_cost_raw > self.cash and buy_cost_raw > 0:
            scale = self.cash / buy_cost_raw
            buy *= scale
            buy = np.floor(buy / self.cfg.lot_size) * self.cfg.lot_size

        buy_value = float((buy * price).sum())
        sell_value = float((sell * price).sum())
        commission_buy = max(buy_value * self.cfg.fee_rate, self.cfg.min_commission if buy_value > 0 else 0)
        commission_sell = max(sell_value * self.cfg.fee_rate, self.cfg.min_commission if sell_value > 0 else 0)
        stamp_tax = sell_value * self.cfg.stamp_tax
        transfer_fee = (buy_value + sell_value) * self.cfg.transfer_fee
        total_cost = commission_buy + commission_sell + stamp_tax + transfer_fee

        # 应用滑点：买单多付，卖单少收
        buy_value_real = buy_value * (1 + slip)
        sell_value_real = sell_value * (1 - slip)

        self.shares = self.shares + buy - sell
        self.cash = (self.cash - buy_value_real - commission_buy - transfer_fee / 2
                     + sell_value_real - commission_sell - stamp_tax - transfer_fee / 2)

        self._bought_today = buy.copy()
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
                    "industry": ev.industry,
                    "direction": ev.direction,
                })
        return out

    def _obs(self) -> dict:
        window = self.cfg.window
        lo = max(0, self.t - window + 1)
        chunk = self.prices[lo:self.t + 1]
        # padding：前 window-1 天历史不足，用首日价格向前填充，
        # 保证 prices.shape 永远是 (window, n_stocks)
        if chunk.shape[0] < window:
            pad = np.tile(chunk[0:1], (window - chunk.shape[0], 1))
            chunk = np.vstack([pad, chunk])
        equity = self.equity
        pos_val = float((self.shares * self.price).sum())
        return {
            "t": self.t,
            "n_days": self.n_days,
            "n_stocks": self.n_stocks,
            "stock_names": self.stock_names,
            "prices": chunk.copy(),
            "last_prices": self.price.copy(),
            "cash": self.cash,
            "shares": self.shares.copy(),
            "equity": equity,
            "position_ratio": pos_val / equity if equity > 0 else 0.0,
            "news": self._recent_news(),
        }


# ============================================================
# 4. Agent 基类 + 基线 + 回测（v4 适配版）
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
    def act(self, obs):
        w = self.rng.random(self.n)
        s = w.sum()
        return w / s if s > 0 else w


class MomentumAgent(Agent):
    def __init__(self, n: int = 1, window: int = 20):
        self.n = n; self.window = window
    def act(self, obs):
        p = obs["prices"]
        if p.shape[0] < 2: return np.zeros(self.n)
        h = p[-min(self.window, p.shape[0]):]
        sig = (h[-1] / h[0] - 1 > 0).astype(float)
        return sig / sig.sum() if sig.sum() > 0 else np.zeros(self.n)


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
    env = TradingEnv(market, env_cfg)
    agent.reset()
    obs = env.reset()
    done = False
    while not done:
        if live:
            t = obs["t"]
            news = obs.get("news", [])
            news_str = " | ".join(n["headline"] for n in news) if news else "-"
            print(f"[Day {t:3d}/{obs['n_days']-1}] 资产={obs['equity']:>12,.2f}  "
                  f"仓位={obs['position_ratio']*100:5.1f}%  新闻: {news_str[:50]}")
        a = agent.act(obs)
        if live:
            print(f"   → 决策: {np.array2string(np.asarray(a), precision=3, threshold=10)}")
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


# ============================================================
# 5. 行业权重（按真实 A 股申万一级市值占比映射）
# ============================================================

INDUSTRY_WEIGHTS = {
    "半导体": 0.173,
    "银行":   0.115,
    "AI算力": 0.097,
    "新能源": 0.071,
    "消费":   0.060,
    "券商":   0.053,
    "医药":   0.046,
    "白酒":   0.028,
    "军工":   0.025,
    "地产":   0.025,
}
