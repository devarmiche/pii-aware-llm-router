import os
from dataclasses import dataclass

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b-instruct"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


def call_local(prompt: str) -> LLMResponse:
    response = requests.post(
        OLLAMA_URL,
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return LLMResponse(
        text=data["response"],
        input_tokens=data["prompt_eval_count"],
        output_tokens=data["eval_count"],
    )


def call_api(prompt: str, model: str) -> LLMResponse:
    """Call a model through OpenRouter (see claude.md for why OpenRouter
    rather than the Mistral API directly)."""
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set — export it before routing to the api backend"
        )

    response = requests.post(
        OPENROUTER_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "messages": [{"role": "user", "content": prompt}]},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    usage = data["usage"]
    return LLMResponse(
        text=data["choices"][0]["message"]["content"],
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
    )
