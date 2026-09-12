import pytest

from src import backends


def test_call_api_raises_clearly_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        backends.call_api("prompt", "some-model")


def test_call_api_parses_openrouter_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")

    captured = {}

    class FakeResponse:
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
