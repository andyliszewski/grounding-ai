"""Claude Messages API calls for the answer benchmark (Epic 25, D3 and D6).

Everything that touches the API goes through :func:`call_model`, which takes
an injectable ``client`` exposing ``client.messages.create(**params)``. Tests
pass a fake client with scripted responses; real runs build one with
:func:`make_anthropic_client`, which reads the key only from the
``ANTHROPIC_API_KEY`` environment variable and never logs it.

Design notes, following the ``claude-api`` skill:

* The SDK's own retries are disabled (``max_retries=0``) and retried here with
  jittered exponential backoff, so each record carries its attempt count and
  the latency of the successful attempt only.
* Adaptive thinking with summarized display and an explicit effort level are
  pinned for models that support them, so a change in API defaults cannot
  silently change a run. Summaries make transcripts auditable; they are never
  shown to a judge.
* The response's ``model`` must match the requested model (or a dated snapshot
  of it). A mismatch fails the answer loudly instead of scoring a substitute.
* No server-side refusal fallback is configured, on purpose: a fallback would
  answer with a different model than the one under test. Refusals are recorded
  as their own outcome.
"""
from __future__ import annotations

import logging
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

logger = logging.getLogger("grounding.eval.answers.model_client")

ANSWER_MAX_TOKENS = 16000
JUDGE_MAX_TOKENS = 8192
EFFORT = "high"
MAX_ATTEMPTS = 4
CLIENT_TIMEOUT_S = 600.0

# Models that take thinking={"type": "adaptive"} and output_config.effort.
# Older models (for example claude-haiku-4-5) get neither and run without
# thinking; the run manifest records exactly what was sent.
_ADAPTIVE_THINKING_PREFIXES = (
    "claude-fable-5",
    "claude-mythos-5",
    "claude-opus-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
)
_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
)


class HarnessError(RuntimeError):
    """An attempt that produced no scorable output (logged to errors.jsonl)."""

    def __init__(
        self,
        failure_class: str,
        message: str,
        *,
        model: str | None = None,
        usage: Dict[str, int] | None = None,
        attempts: int = 0,
    ) -> None:
        super().__init__(message)
        self.failure_class = failure_class
        self.model = model
        self.usage = usage or {}
        self.attempts = attempts


@dataclass
class CallRecord:
    purpose: str
    requested_model: str
    model: str
    stop_reason: str | None
    usage: Dict[str, int]
    latency_s: float
    attempts: int
    content: List[Dict[str, Any]] = field(default_factory=list)
    request_id: str | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "purpose": self.purpose,
            "requested_model": self.requested_model,
            "model": self.model,
            "stop_reason": self.stop_reason,
            "usage": dict(self.usage),
            "latency_s": round(self.latency_s, 4),
            "attempts": self.attempts,
            "request_id": self.request_id,
            "content": self.content,
        }


def supports_adaptive_thinking(model: str) -> bool:
    return model.startswith(_ADAPTIVE_THINKING_PREFIXES)


def generation_options(model: str) -> Dict[str, Any]:
    """Thinking and effort parameters pinned for ``model`` (recorded per run)."""
    if not supports_adaptive_thinking(model):
        return {}
    return {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": EFFORT},
    }


def served_model_matches(requested: str, served: str | None) -> bool:
    if not served:
        return False
    if served == requested:
        return True
    suffix = served[len(requested) + 1 :] if served.startswith(requested + "-") else ""
    return suffix.isdigit()


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


#: Public alias: read ``key`` from a dict, an SDK model, or a simple object.
get_field = _get


def normalize_usage(usage: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for key in _USAGE_KEYS:
        value = _get(usage, key, 0) if usage is not None else 0
        try:
            out[key] = int(value or 0)
        except (TypeError, ValueError):
            out[key] = 0
    return out


def block_to_dict(block: Any) -> Dict[str, Any]:
    """JSON-ready copy of a content block (SDK model, dict, or simple object)."""
    if isinstance(block, dict):
        return dict(block)
    for method in ("model_dump", "to_dict"):
        fn = getattr(block, method, None)
        if callable(fn):
            try:
                data = fn(mode="json", exclude_none=True) if method == "model_dump" else fn()
            except TypeError:
                data = fn()
            if isinstance(data, dict):
                return data
    return {
        key: value
        for key, value in vars(block).items()
        if not key.startswith("_") and value is not None
    }


def response_text(response: Any) -> str:
    """Concatenate the text blocks of a response (thinking is excluded)."""
    parts = []
    for block in _get(response, "content", []) or []:
        if _get(block, "type") == "text":
            parts.append(_get(block, "text", "") or "")
    return "".join(parts).strip()


def tool_use_blocks(response: Any) -> List[Any]:
    return [b for b in (_get(response, "content", []) or []) if _get(b, "type") == "tool_use"]


def _is_retryable(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS
    names = {cls.__name__ for cls in type(exc).__mro__}
    return bool(names & {"APIConnectionError", "APITimeoutError"})


def _failure_class(exc: BaseException) -> str:
    names = {cls.__name__ for cls in type(exc).__mro__}
    if "APITimeoutError" in names:
        return "timeout"
    if getattr(exc, "status_code", None) == 429:
        return "rate_limited"
    return "api_error"


def call_model(
    client: Any,
    params: Dict[str, Any],
    *,
    purpose: str,
    max_attempts: int = MAX_ATTEMPTS,
    sleep: Callable[[float], None] = time.sleep,
    jitter: Callable[[], float] = random.random,
) -> Tuple[Any, CallRecord]:
    """One Messages API call with retries and a served-model check.

    Raises:
        HarnessError: non-retryable API error, retries exhausted, or a
            response served by a different model than requested.
    """
    requested = params["model"]
    attempt = 0
    while True:
        attempt += 1
        start = time.perf_counter()
        try:
            response = client.messages.create(**params)
        except Exception as exc:  # SDK exceptions are not imported on purpose
            if _is_retryable(exc) and attempt < max_attempts:
                delay = min(2.0 ** attempt + jitter(), 30.0)
                logger.warning(
                    "%s call failed (%s), retry %d/%d in %.1fs",
                    purpose,
                    type(exc).__name__,
                    attempt,
                    max_attempts - 1,
                    delay,
                )
                sleep(delay)
                continue
            raise HarnessError(
                _failure_class(exc),
                f"{purpose} call failed after {attempt} attempt(s): {type(exc).__name__}: {exc}",
                model=requested,
                attempts=attempt,
            ) from exc
        latency = time.perf_counter() - start
        break

    served = _get(response, "model")
    usage = normalize_usage(_get(response, "usage"))
    if not served_model_matches(requested, served):
        raise HarnessError(
            "served_model_mismatch",
            f"{purpose} call asked for {requested} but was served by {served}",
            model=served,
            usage=usage,
            attempts=attempt,
        )
    record = CallRecord(
        purpose=purpose,
        requested_model=requested,
        model=served,
        stop_reason=_get(response, "stop_reason"),
        usage=usage,
        latency_s=latency,
        attempts=attempt,
        content=[block_to_dict(b) for b in (_get(response, "content", []) or [])],
        request_id=_get(response, "_request_id") or _get(response, "id"),
    )
    return response, record


def make_anthropic_client(env: Dict[str, str] | None = None):
    """Build a real SDK client from ``ANTHROPIC_API_KEY`` (never logged).

    Raises:
        RuntimeError: no key in the environment, or the SDK is not installed.
    """
    env = os.environ if env is None else env
    api_key = env.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set; export it to run against the API "
            "(use --dry-run to estimate cost without a key)"
        )
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on install
        raise RuntimeError(
            "the answer benchmark needs the anthropic SDK: pip install -e '.[bench]'"
        ) from exc
    return anthropic.Anthropic(
        api_key=api_key, max_retries=0, timeout=CLIENT_TIMEOUT_S
    )


def sdk_version() -> str | None:
    try:
        import anthropic
    except ImportError:
        return None
    return getattr(anthropic, "__version__", None)
