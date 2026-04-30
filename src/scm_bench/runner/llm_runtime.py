"""Shared LLM inference runtime for the LLM baseline team.

A single `LLMRuntime` is held by all four LLM baseline agents in a class
run. Each tick the harness's `pre_tick_callback` calls
`runtime.batch_decide(prompts)` with all four tiers' prompts in one
shot — that is where the v1 ~4× speedup came from (one
`model.generate` over a left-padded batch instead of four sequential
calls).

Three backends share a tiny interface:

- `TransformersBackend` — `AutoTokenizer` + `AutoModelForCausalLM`
  with left-padded batched generate. The reference implementation;
  works on any transformers install.
- `VLLMBackend` — `vllm.LLM(model=...).generate(prompts, sampling_params)`.
  PagedAttention + continuous batching pay off across the 4 tiers ×
  ~200 ticks × N seeds in a class run.
- `OpenAICompatBackend` — HTTP `POST /v1/chat/completions` against any
  OpenAI-compatible endpoint. Default target is Ollama at
  `http://localhost:11434/v1`, which lets a agent point the bench at
  a locally-installed `gemma4:e4b` (or `:e2b`, `:26b`, `:31b`) and
  develop on their laptop. Same shape works for LM Studio, vLLM-serve,
  llama.cpp server, OpenRouter, OpenAI itself. No batching at the wire
  level — Ollama serves prompts sequentially — but the `LLMRuntime`
  surface is unchanged so the rest of the harness doesn't care.

Backend selection follows `SCB_LLM_BACKEND={transformers,vllm,
openai_compat,ollama}` from the environment. Default: `vllm` if
importable, else `transformers`. `ollama` is an alias for
`openai_compat` with the Ollama default URL.

Env vars consumed by `OpenAICompatBackend`:

- `SCB_LLM_BASE_URL`     — endpoint root (default `http://localhost:11434/v1`)
- `SCB_LLM_API_KEY`      — bearer token if the endpoint needs one
- `SCB_LLM_MODEL`        — model id passed to the endpoint
- `SCB_LLM_TIMEOUT_S`    — per-request timeout (default 60)

Tests use a `MockBackend` that returns canned outputs; the
HTTP backend has its own tests against a fake `BaseHTTPRequestHandler`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — imported lazily inside backends
    from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class GenerationResult:
    """One backend output: the generated text and its token count."""

    text: str
    tokens_used: int


class LLMBackend(ABC):
    """The minimal surface every backend must expose."""

    @abstractmethod
    def generate(self, prompts: list[str]) -> list[GenerationResult]:
        """Generate one completion per prompt. Order is preserved."""

    @property
    @abstractmethod
    def model_id(self) -> str: ...


class TransformersBackend(LLMBackend):
    """HuggingFace transformers backend — left-padded batched generate.

    Standard left-padded batched generate. The token-count formula
    `len(output[i][input_ids.shape[1]:])` is fixed to keep numbers
    comparable across runs.
    """

    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int = 48,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover — only on missing deps
            raise RuntimeError(
                "transformers + torch required for TransformersBackend"
            ) from e

        self._model_id = model_id
        self._max_new_tokens = max_new_tokens
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16 if self._device == "cuda" else torch.float32,
            device_map=self._device,
        )
        self._model.eval()

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, prompts: list[str]) -> list[GenerationResult]:
        torch = self._torch
        enc = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(self._device)

        with torch.no_grad():
            out = self._model.generate(
                **enc,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        results: list[GenerationResult] = []
        input_len = enc["input_ids"].shape[1]
        for i in range(len(prompts)):
            new_tokens = out[i][input_len:]
            text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            results.append(GenerationResult(text=text, tokens_used=int(len(new_tokens))))
        return results


class VLLMBackend(LLMBackend):
    """vLLM backend — PagedAttention + continuous batching.

    One `llm.generate(prompts, params)` call returns per-prompt
    `RequestOutput` objects whose `outputs[0].token_ids` length is the
    canonical token count. Greedy sampling (`temperature=0`) keeps
    parity with the transformers backend's `do_sample=False`.
    """

    def __init__(
        self,
        *,
        model_id: str,
        max_new_tokens: int = 48,
        gpu_memory_utilization: float = 0.85,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as e:  # pragma: no cover — vLLM not installed
            raise RuntimeError("vllm not installed; cannot construct VLLMBackend") from e

        self._model_id = model_id
        self._llm = LLM(
            model=model_id,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
        )
        self._sampling = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    def generate(self, prompts: list[str]) -> list[GenerationResult]:
        outputs = self._llm.generate(prompts, self._sampling)
        # vLLM may reorder by request id; sort back to input order using the
        # `request_id` index it assigned, but since `generate(list, params)`
        # preserves input order in current vLLM versions, we rely on that.
        results: list[GenerationResult] = []
        for out in outputs:
            primary = out.outputs[0]
            results.append(
                GenerationResult(
                    text=primary.text,
                    tokens_used=int(len(primary.token_ids)),
                )
            )
        return results


DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


class OpenAICompatBackend(LLMBackend):
    """OpenAI-compatible /v1/chat/completions backend over HTTP.

    Targets any endpoint that speaks the Chat Completions shape — Ollama
    (default), LM Studio, vLLM-serve, llama.cpp server, OpenRouter,
    OpenAI itself. Per-prompt sequential calls; the `LLMRuntime` surface
    accepts that and just iterates. Uses `urllib` so no new dependency.
    """

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        api_key: str | None = None,
        max_new_tokens: int = 48,
        timeout_s: float = 60.0,
    ) -> None:
        self._model_id = model_id
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._max_new_tokens = max_new_tokens
        self._timeout_s = timeout_s

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def base_url(self) -> str:
        return self._base_url

    def generate(self, prompts: list[str]) -> list[GenerationResult]:
        return [self._one(p) for p in prompts]

    def _one(self, prompt: str) -> GenerationResult:
        import json
        import urllib.error
        import urllib.request

        body = json.dumps(
            {
                "model": self._model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self._max_new_tokens,
                "temperature": 0.0,
                "stream": False,
            }
        ).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"OpenAI-compat call to {self._base_url} failed: {e}"
            ) from e

        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError(
                f"OpenAI-compat response missing 'choices': {payload!r}"
            )
        text = choices[0].get("message", {}).get("content", "") or ""
        usage = payload.get("usage") or {}
        tokens = int(usage.get("completion_tokens", 0))
        return GenerationResult(text=text, tokens_used=tokens)


def _resolve_backend(
    *, backend_name: str | None, model_id: str, **kwargs: Any
) -> LLMBackend:
    name = (backend_name or os.environ.get("SCB_LLM_BACKEND") or "auto").lower()
    if name == "transformers":
        return TransformersBackend(model_id=model_id, **kwargs)
    if name == "vllm":
        return VLLMBackend(model_id=model_id, **kwargs)
    if name in ("openai_compat", "ollama"):
        env_kwargs: dict[str, Any] = {}
        if "base_url" not in kwargs:
            env_kwargs["base_url"] = (
                os.environ.get("SCB_LLM_BASE_URL") or DEFAULT_OLLAMA_BASE_URL
            )
        if "api_key" not in kwargs and os.environ.get("SCB_LLM_API_KEY"):
            env_kwargs["api_key"] = os.environ["SCB_LLM_API_KEY"]
        if "timeout_s" not in kwargs and os.environ.get("SCB_LLM_TIMEOUT_S"):
            env_kwargs["timeout_s"] = float(os.environ["SCB_LLM_TIMEOUT_S"])
        return OpenAICompatBackend(model_id=model_id, **env_kwargs, **kwargs)
    if name == "auto":
        try:
            import vllm  # noqa: F401
            return VLLMBackend(model_id=model_id, **kwargs)
        except ImportError:
            return TransformersBackend(model_id=model_id, **kwargs)
    raise ValueError(f"unknown SCB_LLM_BACKEND={name!r}")


class LLMRuntime:
    """A loaded model + tokenizer wrapper, shared across the 4 LLM agents.

    Construct once per batch-run (or once per cluster cell). Every tick
    the LLM team's `pre_tick_callback` calls `batch_decide(prompts)`
    with `prompts` keyed by role; the runtime fans out to the backend
    in a single batched `generate` and returns role → GenerationResult.
    """

    def __init__(
        self,
        *,
        model_id: str,
        backend: LLMBackend | None = None,
        backend_name: str | None = None,
        max_new_tokens: int = 48,
    ) -> None:
        self._model_id = model_id
        if backend is not None:
            self._backend = backend
        else:
            self._backend = _resolve_backend(
                backend_name=backend_name,
                model_id=model_id,
                max_new_tokens=max_new_tokens,
            )

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def backend(self) -> LLMBackend:
        return self._backend

    def batch_decide(self, prompts: dict[str, str]) -> dict[str, GenerationResult]:
        """Run one batched generate across all roles in `prompts`.

        Order of `prompts.keys()` is preserved on the wire so the
        backend sees a deterministic batch. Returned dict keys mirror
        the input.
        """
        roles = list(prompts.keys())
        ordered = [prompts[r] for r in roles]
        outs = self._backend.generate(ordered)
        if len(outs) != len(roles):
            raise RuntimeError(
                f"backend returned {len(outs)} outputs for {len(roles)} prompts"
            )
        return {role: outs[i] for i, role in enumerate(roles)}


__all__ = [
    "GenerationResult",
    "LLMBackend",
    "LLMRuntime",
    "TransformersBackend",
    "VLLMBackend",
]
