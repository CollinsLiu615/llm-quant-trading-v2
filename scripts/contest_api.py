# scripts/contest_api.py
"""LLM 调用封装：超时 + 重试 + 落盘 + 代码提取。"""
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
from models_v4 import MODELS_V4, ModelEndpoint
from code_agent import extract_code, check_code_safety, SecurityError
import contest_config as cfg


def _merge_extra(ep: ModelEndpoint) -> dict:
    """合并 models_v4 的 extra 和 contest_config 的 overrides。"""
    extra = dict(ep.extra)
    overrides = cfg.MODEL_OVERRIDES.get(ep.name, {})
    extra.update(overrides)
    return extra


def _call_with_hard_timeout(fn, timeout, *args, **kwargs):
    """用 daemon 线程实现硬超时，绕过 openai 客户端 timeout 失效。"""
    result = {"value": None, "error": None, "done": False}

    def _run():
        try:
            result["value"] = fn(*args, **kwargs)
        except Exception as e:
            result["error"] = e
        finally:
            result["done"] = True

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if not result["done"]:
        raise TimeoutError(f"hard timeout {timeout}s (client timeout 失效)")
    if result["error"] is not None:
        raise result["error"]
    return result["value"]


def call_llm(model_name: str, prompt: str,
             log_dir: Optional[Path] = None) -> Tuple[Optional[str], str]:
    """
    调一次 API。返回 (返回文本, 状态字符串)。
    状态: "ok" / "api_fail" / "no_key"
    """
    if model_name not in MODELS_V4:
        return None, "unknown_model"
    ep = MODELS_V4[model_name]

    key = os.environ.get(ep.api_key_env, "")
    if not key:
        return None, f"no_key:{ep.api_key_env}"

    extra = _merge_extra(ep)
    client = OpenAI(base_url=ep.base_url, api_key=key, timeout=cfg.API_TIMEOUT)

    for attempt in range(cfg.API_RETRY + 1):
        try:
            t0 = time.time()
            resp = _call_with_hard_timeout(
                client.chat.completions.create,
                cfg.API_TIMEOUT + 30,
                model=ep.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                **extra,
            )
            dt = time.time() - t0
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            tok_in = usage.prompt_tokens if usage else 0
            tok_out = usage.completion_tokens if usage else 0

            # 落盘原始响应
            if log_dir is not None:
                log_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                log_file = log_dir / f"{model_name}_{ts}.txt"
                log_file.write_text(
                    f"# model: {model_name}\n"
                    f"# model_id: {ep.model_id}\n"
                    f"# time: {dt:.1f}s\n"
                    f"# tokens: in={tok_in} out={tok_out}\n"
                    f"# ---PROMPT---\n{prompt}\n# ---RESPONSE---\n{text}\n",
                    encoding="utf-8")
            return text, "ok"
        except Exception as e:
            if attempt < cfg.API_RETRY:
                time.sleep(3)
                continue
            return None, f"api_fail:{type(e).__name__}:{str(e)[:80]}"
    return None, "exhausted"


def call_llm_for_code(model_name: str, prompt: str,
                      n_stocks: int,
                      log_dir: Optional[Path] = None) -> Tuple[Optional[str], str, str]:
    """
    调 API 并提取+校验代码。
    返回 (代码, 状态, 错误信息)。
    状态: "ok" / "extract_fail" / "safety_fail" / "api_fail"
    """
    text, status = call_llm(model_name, prompt, log_dir)
    if status != "ok" or text is None:
        return None, "api_fail", status

    code = extract_code(text)
    if code is None:
        return None, "extract_fail", "响应里找不到 Python 代码块"

    try:
        check_code_safety(code)
    except SecurityError as e:
        return None, "safety_fail", str(e)[:200]

    return code, "ok", ""


if __name__ == "__main__":
    # 自测：对一个便宜模型发一次真实请求
    import sys as _sys
    name = _sys.argv[1] if len(_sys.argv) > 1 else "GLM-4-9B"
    print(f"测试调用 {name} ...")
    text, status = call_llm(name, "回复两字：可用")
    print(f"状态: {status}")
    if text:
        print(f"返回前 100 字: {text[:100]!r}")