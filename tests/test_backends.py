import pytest
import requests

from src import backends


def test_call_local_sizes_num_ctx_to_prompt_length(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"response": "réponse", "prompt_eval_count": 10, "eval_count": 5}

    def fake_post(url: str, json: dict, timeout: int) -> FakeResponse:
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(backends.requests, "post", fake_post)

    backends.call_local("x" * 400)

    assert captured["json"]["options"]["num_ctx"] == 400 // 4 + 512


def test_call_local_caps_num_ctx_at_model_max(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"response": "réponse", "prompt_eval_count": 10, "eval_count": 5}

    def fake_post(url: str, json: dict, timeout: int) -> FakeResponse:
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(backends.requests, "post", fake_post)

    backends.call_local("x" * 1_000_000)

    assert captured["json"]["options"]["num_ctx"] == backends.OLLAMA_MAX_CONTEXT


def test_call_api_raises_clearly_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        backends.call_api("prompt", "some-model")


def test_call_api_parses_openrouter_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    captured = {}

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "choices": [{"message": {"content": "réponse du modèle"}}],
                "usage": {"prompt_tokens": 123, "completion_tokens": 45},
            }

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> FakeResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr(backends.requests, "post", fake_post)

    result = backends.call_api("Quelle est la question ?", "some-model")

    assert result.text == "réponse du modèle"
    assert result.input_tokens == 123
    assert result.output_tokens == 45
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "some-model"
    assert captured["json"]["messages"][0]["content"] == "Quelle est la question ?"


def test_call_api_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(backends.time, "sleep", lambda seconds: None)

    calls = []

    class FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    def fake_post(url: str, headers: dict, json: dict, timeout: int) -> FakeResponse:
        calls.append(1)
        return FakeResponse(429 if len(calls) < 3 else 200)

    monkeypatch.setattr(backends.requests, "post", fake_post)

    result = backends.call_api("prompt", "some-model")

    assert len(calls) == 3
    assert result.text == "ok"


def test_call_api_gives_up_after_max_attempts_on_persistent_429(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(backends.time, "sleep", lambda seconds: None)

    class FakeResponse:
        status_code = 429

        def raise_for_status(self) -> None:
            raise requests.HTTPError("429 Too Many Requests")

    monkeypatch.setattr(backends.requests, "post", lambda *a, **k: FakeResponse())

    with pytest.raises(requests.HTTPError):
        backends.call_api("prompt", "some-model")
