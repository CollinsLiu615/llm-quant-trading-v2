# -*- coding: utf-8 -*-
"""第二阶段参赛模型池（14 个）。"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass
class ModelEndpoint:
    name: str
    base_url: str
    model_id: str
    api_key_env: str
    extra: Dict[str, Any] = field(default_factory=dict)
    cost_per_1k_tokens: float = 0.0


MODELS_V4: Dict[str, ModelEndpoint] = {
    "Qwen3.8-27B": ModelEndpoint(
        name="Qwen3.8-27B",
        base_url="https://chat.ecnu.edu.cn/open/api/v1",
        model_id="ecnu-plus",
        api_key_env="ECNU_API_KEY",
    ),
    "DeepSeek-V4-Flash-0731": ModelEndpoint(
        name="DeepSeek-V4-Flash-0731",
        base_url="https://chat.ecnu.edu.cn/open/api/v1",
        model_id="ecnu-max",
        api_key_env="ECNU_API_KEY",
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
    "Xing4.0-29B-A4B": ModelEndpoint(
        name="Xing4.0-29B-A4B",
        base_url="https://openapi.teleagi.cn/aipaas/xing4-29b/v1",
        model_id="Xing4.0-29B-A4B",
        api_key_env="TELEAI_API_KEY",
        extra={"max_completion_tokens": 4096},
    ),
    "TeleChat3-438B-A24B": ModelEndpoint(
        name="TeleChat3-438B-A24B",
        base_url="https://openapi.teleagi.cn/aipaas/telechat3-438b-a24b/v1",
        model_id="TeleChat3-438B-A24B",
        api_key_env="TELEAI_API_KEY",
        extra={"max_completion_tokens": 4096},
    ),
    "GLM-4-9B": ModelEndpoint(
        name="GLM-4-9B",
        base_url="https://api.siliconflow.cn/v1",
        model_id="THUDM/GLM-4-9B-0414",
        api_key_env="SILICONFLOW_API_KEY",
        extra={"max_tokens": 4096},
    ),
    "GLM-Z1-9B": ModelEndpoint(
        name="GLM-Z1-9B",
        base_url="https://api.siliconflow.cn/v1",
        model_id="THUDM/GLM-Z1-9B-0414",
        api_key_env="SILICONFLOW_API_KEY",
        extra={"max_tokens": 4096},
    ),
    "DeepSeek-R1-8B": ModelEndpoint(
        name="DeepSeek-R1-8B",
        base_url="https://api.siliconflow.cn/v1",
        model_id="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        api_key_env="SILICONFLOW_API_KEY",
        extra={"max_tokens": 4096},
    ),
    "deepseek-v4.1-flash": ModelEndpoint(
        name="deepseek-v4.1-flash",
        base_url="https://api.deepseek.com",
        model_id="deepseek-flash",
        api_key_env="DEEPSEEK_API_KEY",
        extra={"reasoning_effort": "low", "max_tokens": 16384},
        cost_per_1k_tokens=0.025,
    ),
    "deepseek-v4-pro": ModelEndpoint(
        name="deepseek-v4-pro",
        base_url="https://api.deepseek.com",
        model_id="deepseek-v4-pro",
        api_key_env="DEEPSEEK_API_KEY",
        extra={"max_tokens": 16384},
        cost_per_1k_tokens=0.05,
    ),
    "claude-opus-5": ModelEndpoint(
        name="claude-opus-5",
        base_url="https://uuapi.io/v1",
        model_id="claude-opus-5",
        api_key_env="UUAPI_CLAUDE_KEY",
    ),
    "grok-4.7": ModelEndpoint(
        name="grok-4.7",
        base_url="https://uuapi.io/v1",
        model_id="grok-4.7",
        api_key_env="UUAPI_GROK_KEY",
    ),
    "gpt-5.6-sol": ModelEndpoint(
        name="gpt-5.6-sol",
        base_url="https://uuapi.io/v1",
        model_id="gpt-5.6-sol",
        api_key_env="UUAPI_GPT_KEY",
        extra={"max_tokens": 16384},
    ),
    "seed-2.1-turbo": ModelEndpoint(
        name="seed-2.1-turbo",
        base_url="https://codecraftapi.com/v1",
        model_id="seed-2.1-turbo",
        api_key_env="CODECRAFT_API_KEY",
    ),
}


MODELS_ORDERED = [
    "Qwen3.8-27B", "DeepSeek-V4-Flash-0731",
    "glm-5.3-flashx",
    "Xing4.0-29B-A4B", "TeleChat3-438B-A24B",
    "DeepSeek-R1-8B",
    "deepseek-v4.1-flash", "deepseek-v4-pro",
    "claude-opus-5", "grok-4.7", "gpt-5.6-sol", "seed-2.1-turbo",
]