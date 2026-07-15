import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path, override=True)

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "zhipu")

PROVIDER_CONFIG = {
    "gemini": {
        "name": "Google Gemini",
        "api_key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1/models/",
        "default_model": "gemini-2.5-flash",
        "free_tier": True,
        "domestic": False
    },
    "deepseek": {
        "name": "DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "free_tier": True,
        "domestic": True
    },
    "zhipu": {
        "name": "ZhiPu GLM",
        "api_key_env": "ZHIPU_API_KEY",
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-5.2",
        "free_tier": True,
        "domestic": True
    },
    "qianwen": {
        "name": "Aliyun Qianwen",
        "api_key_env": "QIANWEN_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/api/v1",
        "default_model": "qwen-turbo",
        "free_tier": True,
        "domestic": True
    },
    "doubao": {
        "name": "ByteDance Doubao",
        "api_key_env": "DOUBAO_API_KEY",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "default_model": "doubao-pro",
        "free_tier": True,
        "domestic": True
    }
}

def get_current_provider() -> str:
    return LLM_PROVIDER

def get_current_provider_name() -> str:
    return PROVIDER_CONFIG.get(LLM_PROVIDER, {}).get("name", "Unknown")

def get_current_api_key() -> str:
    config = PROVIDER_CONFIG.get(LLM_PROVIDER, {})
    return os.environ.get(config.get("api_key_env", ""), "")

def get_current_model() -> str:
    return PROVIDER_CONFIG.get(LLM_PROVIDER, {}).get("default_model", "")

def get_current_config() -> Dict[str, Any]:
    return PROVIDER_CONFIG.get(LLM_PROVIDER, {})

def is_domestic_provider() -> bool:
    return PROVIDER_CONFIG.get(LLM_PROVIDER, {}).get("domestic", False)