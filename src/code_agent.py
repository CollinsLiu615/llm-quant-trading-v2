# -*- coding: utf-8 -*-
"""CodeAgent：把 AI 生成的 strategy() 代码包装成 Agent。

安全策略：AST 白名单 + 受限 __builtins__ + 编译期校验。
执行期：调用时用 signal.alarm 硬超时。
"""
from __future__ import annotations

import ast
import math
import re
import signal
import sys
from typing import Optional, List

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from stock_sim_v4 import Agent


# ============================================================
# 1. AST 安全检查
# ============================================================

FORBIDDEN_NAMES = {
    "eval", "exec", "compile", "open", "input", "__import__",
    "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr",
    "exit", "quit", "breakpoint", "help",
}

FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib",
    "requests", "urllib", "http", "ftplib", "pickle", "marshal",
    "importlib", "ctypes", "multiprocessing", "threading",
}

ALLOWED_BUILTINS = {
    "len", "sum", "min", "max", "abs", "range", "zip", "sorted",
    "enumerate", "int", "float", "bool", "list", "dict", "tuple",
    "set", "round", "print", "str", "any", "all", "reversed",
    "isinstance", "map", "filter", "pow", "divmod",
}

MAX_CODE_LINES = 250


class SecurityError(Exception):
    pass


class _SecurityVisitor(ast.NodeVisitor):
    """遍历 AST，遇到违规节点就抛 SecurityError。"""

    ALLOWED_IMPORTS = {"numpy", "math"}

    def visit_Import(self, node):
        for a in node.names:
            top = a.name.split(".")[0]
            if top not in self.ALLOWED_IMPORTS:
                raise SecurityError(f"禁止 import: {a.name}")

    def visit_ImportFrom(self, node):
        top = (node.module or "").split(".")[0]
        if top not in self.ALLOWED_IMPORTS:
            raise SecurityError(f"禁止 from-import: {node.module}")

    def visit_Call(self, node):
        # 检查函数名
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_NAMES:
                raise SecurityError(f"禁止调用: {node.func.id}")
        # 检查属性调用（如 os.system）
        if isinstance(node.func, ast.Attribute):
            if node.func.attr.startswith("__"):
                raise SecurityError(f"禁止双下划线属性: {node.func.attr}")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        if node.attr.startswith("__"):
            raise SecurityError(f"禁止双下划线属性: {node.attr}")
        self.generic_visit(node)

    def visit_Name(self, node):
        if node.id in FORBIDDEN_NAMES:
            raise SecurityError(f"禁止名称: {node.id}")
        if node.id in FORBIDDEN_MODULES:
            raise SecurityError(f"禁止模块名: {node.id}")
        self.generic_visit(node)


def check_code_safety(code: str) -> None:
    """对 AI 生成的代码做静态安全检查。
    通过则返回 None，否则抛 SecurityError。
    """
    if not code or not code.strip():
        raise SecurityError("代码为空")

    lines = code.splitlines()
    if len(lines) > MAX_CODE_LINES:
        raise SecurityError(f"代码行数 {len(lines)} 超过上限 {MAX_CODE_LINES}")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SecurityError(f"语法错误: {e}")

    _SecurityVisitor().visit(tree)


def extract_code(text: str) -> Optional[str]:
    """从 AI 回复里抽出 ```python ... ``` 代码块。
    支持无标记（整个文本当代码）和多个代码块（取最长的）。
    """
    if not text:
        return None
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    # 没有 ``` 包裹，尝试整个当代码
    if "def strategy" in text:
        return text.strip()
    return None


# ============================================================
# 2. 受限命名空间 + CodeAgent
# ============================================================

import inspect


def _safe_import(name, *args, **kwargs):
    """替身 __import__：只放行 numpy / math。"""
    top = name.split(".")[0]
    if top == "numpy":
        return np
    if top == "math":
        return math
    raise ImportError(f"模块 {name} 不允许导入")


def _build_namespace() -> dict:
    """构造 exec 时用的受限命名空间。"""
    import builtins as _bi
    safe_builtins = {
        b: getattr(_bi, b) for b in ALLOWED_BUILTINS if hasattr(_bi, b)
    }
    safe_builtins["__import__"] = _safe_import
    return {
        "np": np,
        "numpy": np,
        "math": math,
        "__builtins__": safe_builtins,
    }


class CodeAgent(Agent):
    """把 AI 生成的 strategy() 代码包装成 Agent。

    - 构造时做 AST 安全检查 + 编译 + 提取 strategy 函数
    - 每次 act() 调用 strategy()，带超时保护
    - 任何异常 → 返回空仓，并计入 n_errors
    """

    def __init__(self, code: str, n_stocks: int, timeout: float = 5.0):
        self.code = code
        self.n_stocks = n_stocks
        self.timeout = timeout
        self.n_calls = 0
        self.n_errors = 0
        self.n_parse_failures = 0
        self.last_error: str = ""
        self._strategy = None
        self._compile()

    def _compile(self) -> None:
        check_code_safety(self.code)
        ns = _build_namespace()
        try:
            compiled = compile(self.code, "<ai_strategy>", "exec")
        except SyntaxError as e:
            raise SecurityError(f"编译失败: {e}")
        exec(compiled, ns)
        fn = ns.get("strategy")
        if not callable(fn):
            raise SecurityError("代码里没有可调用的 strategy() 函数")
        # 检查参数个数
        sig = inspect.signature(fn)
        n_params = len([
            p for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ])
        if n_params < 4:
            raise SecurityError(
                f"strategy() 至少需要 4 个位置参数 (prices, news, position, cash)，"
                f"当前 {n_params} 个")
        self._strategy = fn

    def reset(self) -> None:
        self.n_calls = 0
        self.n_errors = 0
        self.last_error = ""

    def act(self, obs: dict) -> np.ndarray:
        self.n_calls += 1
        try:
            prices = obs["prices"]
            news = [n["headline"] for n in obs.get("news", [])]
            position = obs["shares"]
            cash = obs["cash"]
            result = self._call_with_timeout(prices, news, position, cash)
        except Exception as e:
            self.n_errors += 1
            self.n_parse_failures += 1
            self.last_error = f"{type(e).__name__}: {e}"
            return np.zeros(self.n_stocks)

        return self._sanitize(result)

    def _sanitize(self, result) -> np.ndarray:
        """把 strategy 返回的东西转成合法的 weights。"""
        try:
            w = np.asarray(result, dtype=float).reshape(-1)
        except Exception:
            return np.zeros(self.n_stocks)
        if w.size != self.n_stocks:
            return np.zeros(self.n_stocks)
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        w = np.clip(w, 0.0, 1.0)
        s = w.sum()
        if s > 1.0:
            w = w / s
        return w

    def _call_with_timeout(self, *args):
        """用 setitimer 给单次 strategy() 调用加超时。"""
        def _handler(signum, frame):
            raise TimeoutError(f"strategy() 执行超过 {self.timeout}s")

        old = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            return self._strategy(*args)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)


def compile_strategy(code: str, n_stocks: int, timeout: float = 5.0) -> CodeAgent:
    """便捷函数：编译代码并返回 CodeAgent。失败时抛 SecurityError。"""
    return CodeAgent(code, n_stocks, timeout=timeout)
