import os
import time
from dataclasses import dataclass

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral:7b-instruct"
# mistral:7b-instruct's real context window (`ollama show mistral:7b-instruct`).
# Ollama's /api/generate defaults num_ctx to 2048 when it's not set explicitly,
# silently truncating the prompt to that regardless of what the model actually
# supports — confirmed by prompt_eval_count staying ~2051 across documents of
# very different lengths. We size num_ctx to the prompt instead, capped here.
OLLAMA_MAX_CONTEXT = 32768
_CHARS_PER_TOKEN_ESTIMATE = 4  # rough for French/legal text; not exact
_OUTPUT_TOKEN_BUDGET = 512  # num_ctx must also cover the generated answer

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int


def call_local(prompt: str) -> LLMResponse:
    num_ctx = min(
        len(prompt) // _CHARS_PER_TOKEN_ESTIMATE + _OUTPUT_TOKEN_BUDGET,
        OLLAMA_MAX_CONTEXT,
    )
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"num_ctx": num_ctx},
        },
        # Full-context CPU inference on a 7B model is slow — see claude.md.
        timeout=600,
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

    # OpenRouter rate-limits bursts of large requests (429) independently of
    # account credit; retried with backoff rather than surfaced as a failure,
    # since a retry a few seconds later routinely succeeds (see claude.md eval
    # runs). Not retried for other error codes (e.g. 402 payment required).
    max_attempts = 4
    for attempt in range(max_attempts):
        response = requests.post(
            OPENROUTER_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "messages": [{"role": "user", "content": prompt}]},
            timeout=120,
        )
        if response.status_code == 429 and attempt < max_attempts - 1:
            time.sleep(2**attempt * 5)
            continue
        break
    response.raise_for_status()
    data = response.json()
    usage = data["usage"]
    return LLMResponse(
        text=data["choices"][0]["message"]["content"],
        input_tokens=usage["prompt_tokens"],
        output_tokens=usage["completion_tokens"],
    )
