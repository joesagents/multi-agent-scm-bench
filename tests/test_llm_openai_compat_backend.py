"""OpenAICompatBackend HTTP-level tests against a fake server.

The backend is the only path that lets a agent point the bench at a
local Ollama (or any OpenAI-compatible endpoint) without GPU. These
tests stand up a one-shot `BaseHTTPRequestHandler` so we cover wire
behaviour without depending on a running Ollama.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from scm_bench.runner.llm_runtime import (
    DEFAULT_OLLAMA_BASE_URL,
    LLMRuntime,
    OpenAICompatBackend,
    _resolve_backend,
)


class _RecordingHandler(BaseHTTPRequestHandler):
    """Returns a configurable JSON payload and records the request body."""

    payload: dict = {
        "choices": [{"message": {"content": '{"order": 7}'}}],
        "usage": {"completion_tokens": 11},
    }
    captured: list[dict] = []
    response_status: int = 200

    def do_POST(self) -> None:  # noqa: N802 — http.server contract
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        type(self).captured.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "body": json.loads(body) if body else None,
            }
        )
        body_bytes = json.dumps(type(self).payload).encode("utf-8")
        self.send_response(type(self).response_status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)

    def log_message(self, *_a, **_kw) -> None:
        return  # silence test output


@pytest.fixture
def fake_server() -> Iterator[tuple[str, type[_RecordingHandler]]]:
    """Spin up the recording handler on an ephemeral port for one test."""

    class Handler(_RecordingHandler):
        captured: list[dict] = []
        payload: dict = dict(_RecordingHandler.payload)
        response_status: int = 200

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/v1", Handler
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_backend_posts_chat_completions_with_correct_body(fake_server) -> None:
    base_url, Handler = fake_server
    backend = OpenAICompatBackend(model_id="gemma4:e4b", base_url=base_url)

    results = backend.generate(["hello world"])

    assert len(results) == 1
    assert results[0].text == '{"order": 7}'
    assert results[0].tokens_used == 11
    assert len(Handler.captured) == 1
    req = Handler.captured[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["body"]["model"] == "gemma4:e4b"
    assert req["body"]["messages"] == [{"role": "user", "content": "hello world"}]
    assert req["body"]["temperature"] == 0.0
    assert req["body"]["stream"] is False


def test_backend_iterates_over_multiple_prompts(fake_server) -> None:
    base_url, Handler = fake_server
    backend = OpenAICompatBackend(model_id="gemma4:e4b", base_url=base_url)
    prompts = [f"prompt-{i}" for i in range(4)]

    results = backend.generate(prompts)

    assert len(results) == 4
    assert len(Handler.captured) == 4
    seen = [c["body"]["messages"][0]["content"] for c in Handler.captured]
    assert seen == prompts


def test_backend_attaches_bearer_token_when_api_key_set(fake_server) -> None:
    base_url, Handler = fake_server
    backend = OpenAICompatBackend(
        model_id="gemma4:e4b", base_url=base_url, api_key="sk-test"
    )

    backend.generate(["x"])

    auth = Handler.captured[0]["headers"].get("Authorization")
    assert auth == "Bearer sk-test"


def test_backend_omits_authorization_when_no_api_key(fake_server) -> None:
    base_url, Handler = fake_server
    backend = OpenAICompatBackend(model_id="gemma4:e4b", base_url=base_url)
    backend.generate(["x"])
    assert "Authorization" not in Handler.captured[0]["headers"]


def test_backend_returns_zero_tokens_when_usage_missing(fake_server) -> None:
    base_url, Handler = fake_server
    Handler.payload = {"choices": [{"message": {"content": "no usage block"}}]}
    backend = OpenAICompatBackend(model_id="gemma4:e4b", base_url=base_url)

    out = backend.generate(["x"])

    assert out[0].text == "no usage block"
    assert out[0].tokens_used == 0


def test_backend_raises_runtime_error_on_missing_choices(fake_server) -> None:
    base_url, Handler = fake_server
    Handler.payload = {"error": "oops"}
    backend = OpenAICompatBackend(model_id="gemma4:e4b", base_url=base_url)
    with pytest.raises(RuntimeError, match="missing 'choices'"):
        backend.generate(["x"])


def test_backend_raises_runtime_error_on_connection_failure() -> None:
    backend = OpenAICompatBackend(
        model_id="gemma4:e4b",
        base_url="http://127.0.0.1:1",  # nothing listens here
        timeout_s=0.5,
    )
    with pytest.raises(RuntimeError, match="failed"):
        backend.generate(["x"])


def test_resolve_backend_openai_compat_uses_default_ollama_url() -> None:
    backend = _resolve_backend(backend_name="openai_compat", model_id="gemma4:e4b")
    assert isinstance(backend, OpenAICompatBackend)
    assert backend.base_url == DEFAULT_OLLAMA_BASE_URL.rstrip("/")


def test_resolve_backend_ollama_alias_resolves_to_openai_compat() -> None:
    backend = _resolve_backend(backend_name="ollama", model_id="gemma4:e4b")
    assert isinstance(backend, OpenAICompatBackend)


def test_resolve_backend_picks_up_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SCB_LLM_BACKEND", "openai_compat")
    monkeypatch.setenv("SCB_LLM_BASE_URL", "http://example.invalid:9000/v1")
    monkeypatch.setenv("SCB_LLM_API_KEY", "from-env")
    monkeypatch.setenv("SCB_LLM_TIMEOUT_S", "5.5")

    backend = _resolve_backend(backend_name=None, model_id="gemma4:31b")

    assert isinstance(backend, OpenAICompatBackend)
    assert backend.base_url == "http://example.invalid:9000/v1"
    assert backend._api_key == "from-env"
    assert backend._timeout_s == 5.5


def test_runtime_with_openai_compat_backend_round_trip(fake_server) -> None:
    base_url, Handler = fake_server
    backend = OpenAICompatBackend(model_id="gemma4:e4b", base_url=base_url)
    runtime = LLMRuntime(model_id="gemma4:e4b", backend=backend)

    out = runtime.batch_decide(
        {"retailer": "p1", "wholesaler": "p2", "distributor": "p3", "factory": "p4"}
    )

    assert list(out.keys()) == ["retailer", "wholesaler", "distributor", "factory"]
    assert all(o.text == '{"order": 7}' for o in out.values())
    assert len(Handler.captured) == 4
