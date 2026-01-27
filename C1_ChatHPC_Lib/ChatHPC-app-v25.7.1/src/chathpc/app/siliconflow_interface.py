from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chathpc.app.app import AppConfig

import requests
from loguru import logger

from chathpc.app.utils import template_utils


class ChatHPCSiliconFlow:
    def __init__(self, config: AppConfig):
        try:
            logger.info("SILICONFLOW_API_URL is {}", os.environ.get("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1/chat/completions"))
            logger.info("SILICONFLOW_API_KEY set? {}", "SILICONFLOW_API_KEY" in os.environ)
            self.api_url = os.environ.get("SILICONFLOW_API_URL", "https://api.siliconflow.cn/v1/chat/completions")
            self.api_key = os.environ.get("SILICONFLOW_API_KEY")
            if not self.api_key:
                raise ValueError("SILICONFLOW_API_KEY is not set")
        except Exception as e:
            print(
                "Error: Unable to initialize SiliconFlow API client. Please check the settings of SILICONFLOW_API_URL and SILICONFLOW_API_KEY."
            )
            print(e)
            sys.exit(1)
        self.config = config

    def siliconflow_chat_evaluate(self, model_name: str, **kwargs) -> str | None:
        """Evaluate a prompt using SiliconFlow's ChatCompletion API.

        Args:
            model_name (str): Name of the SiliconFlow chat model to use (e.g. 'Pro/deepseek-ai/DeepSeek-V3.2')
            **kwargs: Keyword arguments containing prompt and context
                - prompt: The input prompt to evaluate
                - context: System context/instructions

        Returns:
            str | None: The generated chat response content, stripped of whitespace,
                        or None if chat fails

        Example:
            response = siliconflow_chat_evaluate("Pro/deepseek-ai/DeepSeek-V3.2",
                                                prompt="What is 2+2?",
                                                context="You are a math tutor")
        """

        kw = template_utils.map_keywords(kwargs)

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": kw["context"]},
                    {"role": "user", "content": kw["prompt"]},
                ],
                "max_tokens": self.config.max_response_tokens,
                "temperature": 0.0,
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=180)
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            
            return content
            
        except Exception as e:
            logger.error(f"Error calling SiliconFlow API: {e}")
            return None
