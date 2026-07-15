import os
import time
import json
import requests
from typing import Optional, Dict, List, Any, Union
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(env_path, override=True)
print(f"Loaded .env from: {env_path}")
print(f"LLM_PROVIDER: {os.environ.get('LLM_PROVIDER', 'NOT FOUND')}")
print(f"DEEPSEEK_API_KEY: {os.environ.get('DEEPSEEK_API_KEY', 'NOT FOUND')[:20]}...")

from llm_config import PROVIDER_CONFIG as PROVIDERS, LLM_PROVIDER as DEFAULT_PROVIDER

class LLMProvider:
    def __init__(self, provider: str = None, model: str = None):
        self.provider = provider or DEFAULT_PROVIDER
        self.model = model or PROVIDERS[self.provider]["default_model"]
        self.api_key = os.environ.get(PROVIDERS[self.provider]["api_key_env"], "")
        self.base_url = PROVIDERS[self.provider]["base_url"]
        
        if not self.api_key:
            print(f"Warning: {PROVIDERS[self.provider]['name']} API key not configured")
        
        self.session = requests.Session()
        self.session.verify = False
    
    def call(self, prompt: str, 
             temperature: float = 0.1, 
             max_tokens: int = 2048,
             system_prompt: str = None,
             timeout: int = 60,
             response_format: str = None) -> Optional[str]:
        if not self.api_key:
            return None
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                if self.provider == "gemini":
                    return self._call_gemini(prompt, temperature, max_tokens, timeout)
                elif self.provider == "deepseek":
                    return self._call_openai_compatible(prompt, temperature, max_tokens, system_prompt, timeout, response_format)
                elif self.provider == "zhipu":
                    return self._call_zhipu(prompt, temperature, max_tokens, system_prompt, timeout, response_format)
                elif self.provider == "qianwen":
                    return self._call_openai_compatible(prompt, temperature, max_tokens, system_prompt, timeout, response_format)
                elif self.provider == "doubao":
                    return self._call_openai_compatible(prompt, temperature, max_tokens, system_prompt, timeout, response_format)
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"API call failed ({attempt + 1}/{max_retries}): {e}, retrying...")
                    time.sleep(2)
                else:
                    print(f"API call failed after {max_retries} attempts: {e}")
                    return None
    
    def _call_gemini(self, prompt: str, temperature: float, max_tokens: int, timeout: int) -> Optional[str]:
        url = f"{self.base_url}{self.model}:generateContent?key={self.api_key}"
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        
        response = self.session.post(url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        if "candidates" in result and len(result["candidates"]) > 0:
            content = result["candidates"][0].get("content", {})
            parts = content.get("parts", [])
            if parts:
                return parts[0].get("text", "")
        return None
    
    def _call_openai_compatible(self, prompt: str, temperature: float, max_tokens: int, 
                                system_prompt: str, timeout: int, response_format: str) -> Optional[str]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format == "json":
            data["response_format"] = {"type": "json_object"}
        
        response = self.session.post(url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return None
    
    def _call_zhipu(self, prompt: str, temperature: float, max_tokens: int, 
                    system_prompt: str, timeout: int, response_format: str) -> Optional[str]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        if response_format == "json":
            data["response_format"] = {"type": "json_object"}
        
        response = self.session.post(url, headers=headers, json=data, timeout=timeout)
        response.raise_for_status()
        
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return None

def get_provider(provider: str = None) -> LLMProvider:
    return LLMProvider(provider)

def call_llm(prompt: str, 
             provider: str = None,
             model: str = None,
             temperature: float = 0.1,
             max_tokens: int = 2048,
             system_prompt: str = None,
             timeout: int = 60,
             response_format: str = None) -> Optional[str]:
    print(f"[DEBUG call_llm] Called with provider={provider}, model={model}")
    llm = LLMProvider(provider, model)
    print(f"[DEBUG call_llm] LLMProvider created: provider={llm.provider}, api_key_set={bool(llm.api_key)}")
    result = llm.call(prompt, temperature, max_tokens, system_prompt, timeout, response_format)
    print(f"[DEBUG call_llm] Result: {result[:50] if result else 'None'}")
    return result

def get_available_providers() -> List[str]:
    return list(PROVIDERS.keys())

def get_provider_info(provider: str) -> Dict[str, Any]:
    return PROVIDERS.get(provider, {})