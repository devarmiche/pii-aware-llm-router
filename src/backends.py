from dataclasses import dataclass

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b-instruct"


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
    raise NotImplementedError("OpenRouter not yet configured — see claude.md")
