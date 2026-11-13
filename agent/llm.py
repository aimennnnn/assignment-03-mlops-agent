"""Talks to the language model through the standard chat-completions API.

Works with any provider that offers that API (Groq, Gemini, OpenRouter, Ollama, ...).
Which one is used comes from three settings in .env:
    LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
"""

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

MAX_WAIT = 65  # longest we wait for a rate limit to reset, in seconds


class LLMError(Exception):
    """The language model could not be reached or refused the request."""


class LLM:
    def __init__(self, base_url=None, model=None, api_key=None):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL", "")).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "")
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "")

    def chat(self, messages, tools):
        """Send the conversation and the tool list. Returns the model's reply message."""
        if not (self.base_url and self.model and self.api_key):
            raise LLMError("LLM is not configured: set LLM_BASE_URL, LLM_MODEL and "
                           "LLM_API_KEY in the .env file")
        body = {"model": self.model, "messages": messages, "tools": tools,
                "tool_choice": "auto", "temperature": 0}
        for attempt in range(4):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=body,
                    timeout=90,
                )
            except requests.exceptions.RequestException as e:
                raise LLMError(f"LLM provider is not reachable ({type(e).__name__})") from None
            if r.status_code == 429 and attempt < 3:  # rate limit: wait, then try again
                time.sleep(min(_retry_after(r), MAX_WAIT))
                continue
            if r.status_code >= 500 and attempt < 3:
                time.sleep(2)
                continue
            break
        if r.status_code in (401, 403):
            raise LLMError("LLM provider rejected the API key (check LLM_API_KEY in .env)")
        if r.status_code == 429:
            raise LLMError("LLM rate limit reached; wait a minute and try again")
        if r.status_code >= 400:
            raise LLMError(f"LLM provider answered {r.status_code}: {r.text[:300]}")
        try:
            return r.json()["choices"][0]["message"]
        except (ValueError, KeyError, IndexError):
            raise LLMError("LLM provider sent a reply that could not be read") from None


def _retry_after(response):
    """How many seconds the provider asks us to wait."""
    try:
        return float(response.headers.get("retry-after", 20)) + 1
    except ValueError:
        return 20
