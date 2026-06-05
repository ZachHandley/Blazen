"""Local stdlib-only mock HTTP server for Blazen provider-logic tests.

This module is a *helper*, not a collected test module (its filename starts
with an underscore so pytest does not collect it). It spins up an in-process
HTTP server bound to ``127.0.0.1`` on an OS-assigned free port, so parallel
``pytest -n auto`` workers never share a port or any state.

The server emulates three on-the-wire LLM response formats so the same harness
can drive every Blazen provider whose ``base_url`` is honoured:

* ``"openai"``   -- OpenAI chat-completions JSON / SSE (the OpenAI-compatible
                    family: openai, openrouter, groq, together, mistral,
                    deepseek, fireworks, perplexity, xai, cohere).
* ``"anthropic"``-- Anthropic Messages API JSON (``content`` blocks +
                    ``stop_reason`` + ``usage.input_tokens``).
* ``"gemini"``   -- Google Gemini ``generateContent`` JSON (``candidates`` ->
                    ``content.parts[].text`` + ``usageMetadata``).

It handles **any POST path** so provider-specific URL suffixes
(``/chat/completions``, ``/messages``, ``/v1beta/models/<m>:generateContent``)
all land on the same handler. Every request body is recorded on the controller
so tests can assert request shaping (model name, forwarded messages).

Configurable behaviours, selected per-controller:

* plain JSON chat completion with a chosen ``content`` string;
* streaming SSE for the OpenAI format (several content chunks, a terminal
  chunk carrying ``finish_reason`` + ``usage``, then ``data: [DONE]``);
* error injection -- return a chosen HTTP status (e.g. 429/500/503) with an
  OpenAI-style error JSON, optionally with a ``Retry-After`` header;
* malformed-JSON injection -- 200 OK with a body that is not valid JSON;
* a stateful tool-loop -- the first POST returns an assistant message with a
  ``multiply`` ``tool_calls`` entry (``finish_reason="tool_calls"``); once the
  incoming request body contains a ``role: "tool"`` message, the server returns
  the final assistant content (containing ``"105"``) with ``finish_reason="stop"``.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# ---------------------------------------------------------------------------
# Controller: shared mutable state between the test and the request handler
# ---------------------------------------------------------------------------


@dataclass
class MockController:
    """Holds the configured behaviour and records observed requests.

    A single controller is shared (thread-safely) between the pytest thread
    and the HTTP server's request-handler thread.
    """

    # --- behaviour configuration (set by tests before issuing a request) ---
    wire_format: str = "openai"  # "openai" | "anthropic" | "gemini"
    content: str = "ok"
    model_name: str = "mock-model"
    stream: bool = False
    stream_pieces: list[str] = field(default_factory=lambda: ["Hel", "lo", "!"])
    error_status: int | None = None
    error_body: str = '{"error": {"message": "mock error", "type": "mock"}}'
    retry_after_header: str | None = None
    malformed_json: bool = False
    tool_loop: bool = False

    # --- recorded request state (read by tests after a request) ------------
    last_body: dict[str, Any] | None = None
    last_path: str | None = None
    request_count: int = 0
    saw_tool_result: bool = False

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, path: str, body: dict[str, Any]) -> None:
        with self._lock:
            self.last_path = path
            self.last_body = body
            self.request_count += 1

    def messages_contain_tool_result(self, body: dict[str, Any]) -> bool:
        """True if the request body carries a ``role: "tool"`` message.

        Covers the OpenAI-compatible shape (a top-level ``messages`` array with
        a ``role: "tool"`` entry). The agent loop feeds the tool result back in
        this form for every OpenAI-compatible provider.
        """
        messages = body.get("messages")
        if not isinstance(messages, list):
            return False
        return any(isinstance(m, dict) and m.get("role") == "tool" for m in messages)


# ---------------------------------------------------------------------------
# Response builders (one per wire format)
# ---------------------------------------------------------------------------


def _openai_completion(content: str, model: str, finish_reason: str = "stop") -> bytes:
    payload = {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
    }
    return json.dumps(payload).encode("utf-8")


def _openai_tool_call(model: str) -> bytes:
    """First-turn response: ask to call the ``multiply`` tool."""
    payload = {
        "id": "chatcmpl-mock-tool",
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "multiply",
                                "arguments": '{"a":15,"b":7}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12},
    }
    return json.dumps(payload).encode("utf-8")


def _openai_sse(pieces: list[str], model: str) -> bytes:
    """Build an OpenAI SSE body: content chunks, a terminal chunk, then DONE."""
    lines: list[str] = []
    for piece in pieces:
        chunk = {
            "id": "chatcmpl-mock-stream",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [
                {"index": 0, "delta": {"content": piece}, "finish_reason": None}
            ],
        }
        lines.append(f"data: {json.dumps(chunk)}\n\n")

    terminal = {
        "id": "chatcmpl-mock-stream",
        "object": "chat.completion.chunk",
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
    }
    lines.append(f"data: {json.dumps(terminal)}\n\n")
    lines.append("data: [DONE]\n\n")
    return "".join(lines).encode("utf-8")


def _anthropic_completion(content: str, model: str) -> bytes:
    payload = {
        "id": "msg_mock",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": content}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 7, "output_tokens": 5},
    }
    return json.dumps(payload).encode("utf-8")


def _gemini_completion(content: str, model: str) -> bytes:
    payload = {
        "candidates": [
            {
                "content": {"parts": [{"text": content}], "role": "model"},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 7,
            "candidatesTokenCount": 5,
            "totalTokenCount": 12,
        },
        "modelVersion": model,
    }
    return json.dumps(payload).encode("utf-8")


# ---------------------------------------------------------------------------
# Request handler
# ---------------------------------------------------------------------------


def _make_handler(controller: MockController) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        # Silence the default stderr request logging.
        def log_message(self, *_args: Any) -> None:  # noqa: D401
            return

        def _read_body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw = self.rfile.read(length) if length else b""
            try:
                parsed = json.loads(raw.decode("utf-8")) if raw else {}
            except (ValueError, UnicodeDecodeError):
                parsed = {}
            return parsed if isinstance(parsed, dict) else {}

        def _send(
            self,
            status: int,
            body: bytes,
            content_type: str = "application/json",
            extra_headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for key, value in (extra_headers or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802  (stdlib naming)
            body = self._read_body()
            controller.record(self.path, body)

            # --- error injection -------------------------------------------
            if controller.error_status is not None:
                extra = {}
                if controller.retry_after_header is not None:
                    extra["Retry-After"] = controller.retry_after_header
                self._send(
                    controller.error_status,
                    controller.error_body.encode("utf-8"),
                    extra_headers=extra,
                )
                return

            # --- malformed JSON (200 OK, unparseable body) -----------------
            if controller.malformed_json:
                self._send(200, b"this is { definitely not :: json")
                return

            # --- stateful tool loop (OpenAI wire format only) --------------
            if controller.tool_loop:
                if controller.messages_contain_tool_result(body):
                    controller.saw_tool_result = True
                    self._send(
                        200,
                        _openai_completion(
                            "The product is 105.", controller.model_name
                        ),
                    )
                else:
                    self._send(200, _openai_tool_call(controller.model_name))
                return

            # --- streaming (OpenAI SSE) ------------------------------------
            if controller.stream and controller.wire_format == "openai":
                self._send(
                    200,
                    _openai_sse(controller.stream_pieces, controller.model_name),
                    content_type="text/event-stream",
                )
                return

            # --- plain completion, per wire format -------------------------
            if controller.wire_format == "anthropic":
                self._send(
                    200,
                    _anthropic_completion(controller.content, controller.model_name),
                )
            elif controller.wire_format == "gemini":
                self._send(
                    200,
                    _gemini_completion(controller.content, controller.model_name),
                )
            else:
                self._send(
                    200,
                    _openai_completion(controller.content, controller.model_name),
                )

    return Handler


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


@dataclass
class MockServer:
    """A running mock server plus its controller and base URL."""

    httpd: HTTPServer
    thread: threading.Thread
    controller: MockController
    port: int

    @property
    def base_url(self) -> str:
        """Base URL suitable for ``ProviderOptions(base_url=...)``.

        Includes a ``/v1`` suffix so OpenAI-compatible providers append
        ``/chat/completions`` to a sensible-looking root. The handler accepts
        any POST path regardless, so the suffix is purely cosmetic.
        """
        return f"http://127.0.0.1:{self.port}/v1"

    def shutdown(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


def start_mock_server() -> MockServer:
    """Start a fresh mock server on an OS-assigned free port."""
    controller = MockController()
    handler_cls = _make_handler(controller)
    httpd = HTTPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return MockServer(httpd=httpd, thread=thread, controller=controller, port=port)
