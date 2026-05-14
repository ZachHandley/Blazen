"""Typed-method subclass tests for ``CustomProvider``.

Covers the Phase B-Pivot user-facing surface:

* Subclassing ``CustomProvider`` with an ``async def`` override for a typed
  compute method (``text_to_speech``) routes Python-level calls through the
  override (Python's MRO finds the subclass method before the inherited
  pyclass shim).
* Subclassing with an ``async def complete`` override routes completion
  calls through the override.
* Methods that are *not* overridden surface a ``BlazenError`` (the base
  exception class; specifically, the adapter raises ``UnsupportedError``
  when called via the Rust trait or ``ProviderError`` when the inherited
  pyclass shim re-enters the adapter with a serialized payload).
* The ``ollama`` classmethod factory constructs a usable handle and
  reports ``provider_id == "ollama"`` without performing any network I/O.
* ``BaseProvider.extract(schema, messages)`` returns a pydantic instance
  populated from a stubbed ``complete()`` override.

No API keys or network access are required.
"""

import pytest

from blazen import (
    BlazenError,
    ChatMessage,
    CustomProvider,
    ImageRequest,
    SpeechRequest,
)

# pydantic is required only for the ``extract`` test. It is not a hard
# dependency of the test suite (the Blazen workspace ``pyproject.toml``
# does not list it), so import it lazily inside the test and skip when
# the module is unavailable.
try:
    import pydantic as _pydantic  # noqa: F401
    _HAVE_PYDANTIC = True
except ImportError:  # pragma: no cover
    _HAVE_PYDANTIC = False

# ---------------------------------------------------------------------------
# 1. text_to_speech override routes through the subclass
# ---------------------------------------------------------------------------


_STUB_AUDIO_RESULT = {
    "audio": [],
    "timing": {"queue_ms": None, "execution_ms": None, "total_ms": 0},
    "cost": None,
    "audio_seconds": 0.0,
    "metadata": {"stub": True, "kind": "tts"},
}


class StubTts(CustomProvider):
    """A ``CustomProvider`` subclass that only overrides ``text_to_speech``."""

    def __new__(cls):
        return super().__new__(cls, provider_id="stub-tts")

    def __init__(self) -> None:
        super().__init__(provider_id="stub-tts")
        self.tts_calls = 0
        self.last_text: str | None = None

    async def text_to_speech(self, request):  # type: ignore[override]
        self.tts_calls += 1
        self.last_text = request["text"] if isinstance(request, dict) else request.text
        return _STUB_AUDIO_RESULT


async def test_subclass_text_to_speech_routes_to_override():
    provider = StubTts()
    result = await provider.text_to_speech(SpeechRequest(text="hello world"))

    # Override fired exactly once with the right text.
    assert provider.tts_calls == 1
    assert provider.last_text == "hello world"

    # Python's MRO finds the subclass override before the inherited
    # pyclass shim, so the return value is the exact object the override
    # produced (the typed conversion only runs when Rust framework code
    # drives the provider through the trait).
    assert result is _STUB_AUDIO_RESULT
    assert result["metadata"]["stub"] is True


# ---------------------------------------------------------------------------
# 2. complete override routes through the subclass
# ---------------------------------------------------------------------------


class StubComplete(CustomProvider):
    """A ``CustomProvider`` subclass that only overrides ``complete``."""

    def __new__(cls, content: str = "stub-response"):
        return super().__new__(cls, provider_id="stub-complete")

    def __init__(self, content: str = "stub-response") -> None:
        super().__init__(provider_id="stub-complete")
        self._content = content
        self.complete_calls = 0

    async def complete(self, request):  # type: ignore[override]
        self.complete_calls += 1
        return {
            "content": self._content,
            "model": "stub-complete",
            "tool_calls": [],
            "finish_reason": "stop",
            "metadata": {},
        }


async def test_subclass_complete_routes_to_override():
    provider = StubComplete(content="hi from stub")
    response = await provider.complete([ChatMessage.user("ping")])

    assert provider.complete_calls == 1
    # Direct Python invocation returns the raw object yielded by the
    # override; the typed ``CompletionResponse`` wrapper is only applied
    # when Rust trait dispatch drives the call.
    assert isinstance(response, dict)
    assert response["content"] == "hi from stub"
    assert response["model"] == "stub-complete"
    assert response["finish_reason"] == "stop"


# ---------------------------------------------------------------------------
# 3. Unimplemented methods raise UnsupportedError
# ---------------------------------------------------------------------------


async def test_unimplemented_method_raises_unsupported():
    """A subclass that overrides only ``text_to_speech`` must raise a
    ``BlazenError`` when an unsupported method (``generate_image``) is
    called.

    The exact subclass depends on the dispatch path: the typed Rust
    trait raises ``UnsupportedError`` for missing Python methods, while
    the inherited pyclass shim re-enters the adapter with a serialized
    payload and surfaces ``ProviderError``. Both inherit from
    ``BlazenError``, which is the stable assertion target.
    """
    provider = StubTts()
    with pytest.raises(BlazenError):
        await provider.generate_image(ImageRequest(prompt="a red square"))


# ---------------------------------------------------------------------------
# 4. CustomProvider.ollama classmethod factory
# ---------------------------------------------------------------------------


def test_ollama_factory_constructs_provider():
    """``CustomProvider.ollama`` returns a built provider with the right id.

    Signature: ``ollama(model, host=None, port=None)`` -- ``model`` is the
    first positional argument; defaults are ``localhost:11434``. No network
    call is made; only construction is exercised.
    """
    provider = CustomProvider.ollama("llama3", "localhost", 11434)
    assert isinstance(provider, CustomProvider)
    assert provider.provider_id == "ollama"

    # Same provider built with explicit host/port keywords.
    provider2 = CustomProvider.ollama(model="llama3", host="localhost", port=11434)
    assert provider2.provider_id == "ollama"


# ---------------------------------------------------------------------------
# 5. BaseProvider.extract(schema, messages) -> pydantic instance
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAVE_PYDANTIC, reason="pydantic not installed")
async def test_extract_with_pydantic_model():
    import pydantic

    class Person(pydantic.BaseModel):
        name: str
        age: int

    class StubExtract(CustomProvider):
        """Subclass whose ``complete`` returns a JSON blob shaped like ``Person``."""

        def __new__(cls):
            return super().__new__(cls, provider_id="stub-extract")

        def __init__(self) -> None:
            super().__init__(provider_id="stub-extract")

        async def complete(self, request):  # type: ignore[override]
            return {
                "content": '{"name":"Alice","age":30}',
                "model": "stub-extract",
                "tool_calls": [],
                "finish_reason": "stop",
                "metadata": {},
            }

    provider = StubExtract()
    person = await provider.extract(Person, [ChatMessage.user("who is alice?")])

    assert isinstance(person, Person)
    assert person.name == "Alice"
    assert person.age == 30
