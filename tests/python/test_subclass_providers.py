"""Tests for subclassable provider classes.

Verifies that PyO3 subclass machinery works: Python users can subclass
the base provider types, override methods, and the dispatch works
correctly (unoverridden methods raise NotImplementedError, overridden
methods execute the Python implementation).

None of these tests require API keys or network access.
"""

import pytest

from blazen import (
    BackgroundRemovalProvider,
    ChatMessage,
    CompletionModel,
    CompletionOptions,
    EmbeddingModel,
    ImageProvider,
    InMemoryBackend,
    Memory,
    MemoryBackend,
    ModelManager,
    ModelPricing,
    MusicProvider,
    SpeechRequest,
    TTSProvider,
    ThreeDProvider,
    Transcription,
    TranscriptionRequest,
    VideoProvider,
    VoiceProvider,
    lookup_pricing,
    register_pricing,
)


# ---------------------------------------------------------------------------
# CompletionModel subclass
# ---------------------------------------------------------------------------


class MockLLM(CompletionModel):
    """A mock CompletionModel subclass for testing."""

    def __init__(self):
        super().__init__(model_id="mock-llm", context_length=4096)


def test_subclass_completion_model_instantiation():
    """Subclassing CompletionModel creates a valid instance."""
    model = MockLLM()
    assert model.model_id == "mock-llm"


def test_subclass_completion_model_is_instance():
    """Subclass instance passes isinstance check."""
    model = MockLLM()
    assert isinstance(model, CompletionModel)
    assert isinstance(model, MockLLM)


@pytest.mark.asyncio
async def test_subclass_completion_model_unoverridden_raises():
    """Calling complete() without overriding raises NotImplementedError."""
    model = MockLLM()
    with pytest.raises(NotImplementedError, match="subclass must override complete"):
        await model.complete([ChatMessage.user("hello")])


@pytest.mark.asyncio
async def test_subclass_completion_model_stream_unoverridden_raises():
    """Calling stream() without overriding raises NotImplementedError."""
    model = MockLLM()
    with pytest.raises(NotImplementedError, match="subclass must override stream"):
        model.stream([ChatMessage.user("hello")])


def test_subclass_completion_model_with_optional_params():
    """Constructor keyword arguments are passed through."""

    class DetailedLLM(CompletionModel):
        def __init__(self):
            super().__init__(
                model_id="detailed-llm",
                context_length=128_000,
                base_url="http://localhost:8080",
                max_output_tokens=4096,
            )

    model = DetailedLLM()
    assert model.model_id == "detailed-llm"


def test_subclass_completion_model_with_pricing():
    """Pricing can be attached at construction time."""
    pricing = ModelPricing(input_per_million=1.0, output_per_million=2.0)

    class PricedLLM(CompletionModel):
        def __init__(self):
            super().__init__(model_id="priced-llm", pricing=pricing)

    model = PricedLLM()
    assert model.model_id == "priced-llm"


def test_subclass_completion_model_repr():
    """repr() works on subclassed instances."""
    model = MockLLM()
    r = repr(model)
    assert isinstance(r, str)


# ---------------------------------------------------------------------------
# EmbeddingModel subclass
# ---------------------------------------------------------------------------


class MockEmbedder(EmbeddingModel):
    """A mock EmbeddingModel subclass for testing."""

    def __init__(self):
        super().__init__(model_id="mock-embedder", dimensions=128)


def test_subclass_embedding_model_instantiation():
    """Subclassing EmbeddingModel creates a valid instance."""
    model = MockEmbedder()
    assert model.model_id == "mock-embedder"
    assert model.dimensions == 128


def test_subclass_embedding_model_is_instance():
    """Subclass instance passes isinstance check."""
    model = MockEmbedder()
    assert isinstance(model, EmbeddingModel)
    assert isinstance(model, MockEmbedder)


@pytest.mark.asyncio
async def test_subclass_embedding_model_unoverridden_raises():
    """Calling embed() without overriding raises NotImplementedError."""
    model = MockEmbedder()
    with pytest.raises(NotImplementedError):
        await model.embed(["hello", "world"])


def test_subclass_embedding_model_with_all_params():
    """All optional constructor params work."""

    class DetailedEmbedder(EmbeddingModel):
        def __init__(self):
            super().__init__(
                model_id="detailed-embedder",
                dimensions=768,
                base_url="http://localhost:9090",
                pricing=ModelPricing(input_per_million=0.1),
                vram_estimate_bytes=2_000_000_000,
            )

    model = DetailedEmbedder()
    assert model.model_id == "detailed-embedder"
    assert model.dimensions == 768


# ---------------------------------------------------------------------------
# Transcription subclass
# ---------------------------------------------------------------------------


class MockTranscriber(Transcription):
    """A mock Transcription subclass for testing."""

    def __init__(self):
        super().__init__(provider_id="mock-stt")


def test_subclass_transcription_instantiation():
    """Subclassing Transcription creates a valid instance."""
    transcriber = MockTranscriber()
    assert transcriber.provider_id == "mock-stt"


def test_subclass_transcription_is_instance():
    """Subclass instance passes isinstance check."""
    transcriber = MockTranscriber()
    assert isinstance(transcriber, Transcription)
    assert isinstance(transcriber, MockTranscriber)


@pytest.mark.asyncio
async def test_subclass_transcription_unoverridden_raises():
    """Calling transcribe() without overriding raises NotImplementedError."""
    transcriber = MockTranscriber()
    request = TranscriptionRequest(audio_url="https://example.com/audio.wav")
    with pytest.raises(NotImplementedError):
        await transcriber.transcribe(request)


def test_subclass_transcription_with_all_params():
    """All optional constructor params work."""

    class DetailedTranscriber(Transcription):
        def __init__(self):
            super().__init__(
                provider_id="detailed-stt",
                base_url="http://localhost:7070",
                pricing=ModelPricing(per_second=0.001),
                vram_estimate_bytes=1_000_000_000,
            )

    t = DetailedTranscriber()
    assert t.provider_id == "detailed-stt"


# ---------------------------------------------------------------------------
# TTSProvider subclass
# ---------------------------------------------------------------------------


class MockTTS(TTSProvider):
    """A mock TTS provider subclass for testing."""

    def __init__(self):
        super().__init__(provider_id="mock-tts")


def test_subclass_tts_instantiation():
    """Subclassing TTSProvider creates a valid instance."""
    tts = MockTTS()
    assert tts.provider_id == "mock-tts"


def test_subclass_tts_is_instance():
    """Subclass instance passes isinstance check."""
    tts = MockTTS()
    assert isinstance(tts, TTSProvider)
    assert isinstance(tts, MockTTS)


def test_subclass_tts_unoverridden_raises():
    """Calling text_to_speech() without overriding raises NotImplementedError."""
    tts = MockTTS()
    request = SpeechRequest(text="Hello world")
    with pytest.raises(NotImplementedError):
        tts.text_to_speech(request)


def test_subclass_tts_with_all_params():
    """All optional constructor params work."""

    class DetailedTTS(TTSProvider):
        def __init__(self):
            super().__init__(
                provider_id="detailed-tts",
                base_url="http://localhost:6060",
                pricing=ModelPricing(per_second=0.002),
                vram_estimate_bytes=500_000_000,
            )

    tts = DetailedTTS()
    assert tts.provider_id == "detailed-tts"
    assert tts.base_url == "http://localhost:6060"
    assert tts.vram_estimate_bytes == 500_000_000


# ---------------------------------------------------------------------------
# ImageProvider subclass
# ---------------------------------------------------------------------------


class MockImageProvider(ImageProvider):
    """A mock image provider subclass for testing."""

    def __init__(self):
        super().__init__(provider_id="mock-image")


def test_subclass_image_provider_instantiation():
    """Subclassing ImageProvider creates a valid instance."""
    provider = MockImageProvider()
    assert provider.provider_id == "mock-image"


def test_subclass_image_provider_is_instance():
    """Subclass instance passes isinstance check."""
    provider = MockImageProvider()
    assert isinstance(provider, ImageProvider)
    assert isinstance(provider, MockImageProvider)


# ---------------------------------------------------------------------------
# VideoProvider subclass
# ---------------------------------------------------------------------------


class MockVideoProvider(VideoProvider):
    """A mock video provider subclass for testing."""

    def __init__(self):
        super().__init__(provider_id="mock-video")


def test_subclass_video_provider_instantiation():
    """Subclassing VideoProvider creates a valid instance."""
    provider = MockVideoProvider()
    assert provider.provider_id == "mock-video"


def test_subclass_video_provider_is_instance():
    """Subclass instance passes isinstance check."""
    provider = MockVideoProvider()
    assert isinstance(provider, VideoProvider)
    assert isinstance(provider, MockVideoProvider)


# ---------------------------------------------------------------------------
# MusicProvider subclass
# ---------------------------------------------------------------------------


class MockMusicProvider(MusicProvider):
    """A mock music provider subclass for testing."""

    def __init__(self):
        super().__init__(provider_id="mock-music")


def test_subclass_music_provider_instantiation():
    """Subclassing MusicProvider creates a valid instance."""
    provider = MockMusicProvider()
    assert provider.provider_id == "mock-music"


def test_subclass_music_provider_is_instance():
    """Subclass instance passes isinstance check."""
    provider = MockMusicProvider()
    assert isinstance(provider, MusicProvider)
    assert isinstance(provider, MockMusicProvider)


# ---------------------------------------------------------------------------
# VoiceProvider subclass
# ---------------------------------------------------------------------------


class MockVoiceProvider(VoiceProvider):
    """A mock voice provider subclass for testing."""

    def __init__(self):
        super().__init__(provider_id="mock-voice")


def test_subclass_voice_provider_instantiation():
    """Subclassing VoiceProvider creates a valid instance."""
    provider = MockVoiceProvider()
    assert provider.provider_id == "mock-voice"


def test_subclass_voice_provider_is_instance():
    """Subclass instance passes isinstance check."""
    provider = MockVoiceProvider()
    assert isinstance(provider, VoiceProvider)
    assert isinstance(provider, MockVoiceProvider)


# ---------------------------------------------------------------------------
# ThreeDProvider subclass
# ---------------------------------------------------------------------------


class MockThreeDProvider(ThreeDProvider):
    """A mock 3D provider subclass for testing."""

    def __init__(self):
        super().__init__(provider_id="mock-3d")


def test_subclass_3d_provider_instantiation():
    """Subclassing ThreeDProvider creates a valid instance."""
    provider = MockThreeDProvider()
    assert provider.provider_id == "mock-3d"


def test_subclass_3d_provider_is_instance():
    """Subclass instance passes isinstance check."""
    provider = MockThreeDProvider()
    assert isinstance(provider, ThreeDProvider)
    assert isinstance(provider, MockThreeDProvider)


# ---------------------------------------------------------------------------
# BackgroundRemovalProvider subclass
# ---------------------------------------------------------------------------


class MockBgRemovalProvider(BackgroundRemovalProvider):
    """A mock background removal provider subclass for testing."""

    def __init__(self):
        super().__init__(provider_id="mock-bgrm")


def test_subclass_bg_removal_provider_instantiation():
    """Subclassing BackgroundRemovalProvider creates a valid instance."""
    provider = MockBgRemovalProvider()
    assert provider.provider_id == "mock-bgrm"


def test_subclass_bg_removal_provider_is_instance():
    """Subclass instance passes isinstance check."""
    provider = MockBgRemovalProvider()
    assert isinstance(provider, BackgroundRemovalProvider)
    assert isinstance(provider, MockBgRemovalProvider)


# ---------------------------------------------------------------------------
# MemoryBackend subclass
# ---------------------------------------------------------------------------


class MockMemoryBackend(MemoryBackend):
    """A mock MemoryBackend subclass that stores entries in a dict."""

    def __init__(self):
        super().__init__()
        self._store = {}

    async def put(self, entry):
        entry_id = entry.get("id", "unknown")
        self._store[entry_id] = entry
        return entry_id

    async def get(self, entry_id):
        return self._store.get(entry_id)

    async def delete(self, entry_id):
        return self._store.pop(entry_id, None) is not None

    async def list(self):
        return list(self._store.values())

    async def len(self):
        return len(self._store)

    async def search_by_bands(self, bands, limit):
        # Trivial: return everything up to limit
        return list(self._store.values())[:limit]


def test_subclass_memory_backend_instantiation():
    """Subclassing MemoryBackend creates a valid instance."""
    backend = MockMemoryBackend()
    assert isinstance(backend, MemoryBackend)
    assert isinstance(backend, MockMemoryBackend)


def test_subclass_memory_backend_unoverridden_raises():
    """Base MemoryBackend methods raise NotImplementedError."""

    class BareBackend(MemoryBackend):
        def __init__(self):
            super().__init__()

    backend = BareBackend()
    with pytest.raises(NotImplementedError):
        backend.put({"id": "x", "text": "hello"})


def test_subclass_memory_backend_with_local_memory():
    """A custom MemoryBackend can be passed to Memory.local()."""
    backend = MockMemoryBackend()
    memory = Memory.local(backend)
    assert isinstance(memory, Memory)


@pytest.mark.asyncio
async def test_subclass_memory_backend_add_and_count():
    """Memory.local() with custom backend supports add and count."""
    backend = MockMemoryBackend()
    memory = Memory.local(backend)
    await memory.add("doc1", "Paris is the capital of France")
    count = await memory.count()
    assert count == 1


@pytest.mark.asyncio
async def test_subclass_memory_backend_add_and_search_local():
    """Memory.local() with custom backend supports search_local."""
    backend = MockMemoryBackend()
    memory = Memory.local(backend)
    await memory.add("doc1", "Paris is the capital of France")
    await memory.add("doc2", "Berlin is the capital of Germany")
    results = await memory.search_local("capital", limit=5)
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_subclass_memory_backend_delete():
    """Memory.local() with custom backend supports delete."""
    backend = MockMemoryBackend()
    memory = Memory.local(backend)
    await memory.add("doc1", "Paris is the capital of France")
    deleted = await memory.delete("doc1")
    assert deleted is True
    count = await memory.count()
    assert count == 0


# ---------------------------------------------------------------------------
# InMemoryBackend (built-in, not subclassed -- sanity check)
# ---------------------------------------------------------------------------


def test_in_memory_backend_creation():
    """InMemoryBackend can be created."""
    backend = InMemoryBackend()
    assert isinstance(backend, InMemoryBackend)


def test_in_memory_backend_with_local_memory():
    """InMemoryBackend works with Memory.local()."""
    backend = InMemoryBackend()
    memory = Memory.local(backend)
    assert isinstance(memory, Memory)


@pytest.mark.asyncio
async def test_in_memory_backend_roundtrip():
    """InMemoryBackend add/count/search_local/delete roundtrip."""
    backend = InMemoryBackend()
    memory = Memory.local(backend)

    doc_id = await memory.add("test1", "The quick brown fox")
    assert doc_id == "test1"

    count = await memory.count()
    assert count == 1

    results = await memory.search_local("fox", limit=5)
    assert len(results) >= 1
    assert results[0].text == "The quick brown fox"

    deleted = await memory.delete("test1")
    assert deleted is True

    count = await memory.count()
    assert count == 0


# ---------------------------------------------------------------------------
# ModelPricing + pricing functions
# ---------------------------------------------------------------------------


def test_model_pricing_creation():
    """ModelPricing can be created with various cost fields."""
    pricing = ModelPricing(
        input_per_million=2.50,
        output_per_million=10.00,
        per_image=0.01,
        per_second=0.005,
    )
    assert pricing.input_per_million == 2.50
    assert pricing.output_per_million == 10.00
    assert pricing.per_image == 0.01
    assert pricing.per_second == 0.005


def test_model_pricing_partial():
    """ModelPricing works with only some fields set."""
    pricing = ModelPricing(input_per_million=1.0)
    assert pricing.input_per_million == 1.0
    assert pricing.output_per_million is None
    assert pricing.per_image is None
    assert pricing.per_second is None


def test_model_pricing_repr():
    """ModelPricing has a working repr."""
    pricing = ModelPricing(input_per_million=1.0, output_per_million=2.0)
    r = repr(pricing)
    assert isinstance(r, str)


def test_pricing_register_and_lookup():
    """register_pricing and lookup_pricing work for custom models."""
    pricing = ModelPricing(input_per_million=1.5, output_per_million=3.0)
    register_pricing("test-custom-model-12345", pricing)

    result = lookup_pricing("test-custom-model-12345")
    assert result is not None
    assert result.input_per_million == 1.5
    assert result.output_per_million == 3.0


def test_lookup_pricing_unknown_model():
    """lookup_pricing returns None for unknown models."""
    result = lookup_pricing("definitely-not-a-real-model-xyz-999")
    assert result is None


def test_pricing_register_overwrite():
    """Registering pricing for the same model ID overwrites."""
    register_pricing(
        "test-overwrite-model-67890",
        ModelPricing(input_per_million=1.0, output_per_million=2.0),
    )
    register_pricing(
        "test-overwrite-model-67890",
        ModelPricing(input_per_million=5.0, output_per_million=10.0),
    )

    result = lookup_pricing("test-overwrite-model-67890")
    assert result is not None
    assert result.input_per_million == 5.0
    assert result.output_per_million == 10.0


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------


def test_model_manager_creation_gb():
    """ModelManager can be created with a GB budget."""
    manager = ModelManager(budget_gb=24.0)
    assert isinstance(manager, ModelManager)


def test_model_manager_creation_bytes():
    """ModelManager can be created with a byte budget."""
    manager = ModelManager(budget_bytes=24 * 1024 * 1024 * 1024)
    assert isinstance(manager, ModelManager)


def test_model_manager_creation_no_budget():
    """ModelManager can be created without a budget (unlimited)."""
    manager = ModelManager()
    assert isinstance(manager, ModelManager)


@pytest.mark.asyncio
async def test_model_manager_status_empty():
    """A fresh ModelManager has no registered models."""
    manager = ModelManager(budget_gb=8.0)
    status = await manager.status()
    assert isinstance(status, list)
    assert len(status) == 0


@pytest.mark.asyncio
async def test_model_manager_used_bytes_empty():
    """A fresh ModelManager uses 0 bytes."""
    manager = ModelManager(budget_gb=8.0)
    used = await manager.used_bytes()
    assert used == 0


@pytest.mark.asyncio
async def test_model_manager_available_bytes():
    """available_bytes equals the full budget when nothing is loaded."""
    budget_bytes = 8 * 1024 * 1024 * 1024
    manager = ModelManager(budget_bytes=budget_bytes)
    available = await manager.available_bytes()
    assert available == budget_bytes


@pytest.mark.asyncio
async def test_model_manager_is_loaded_nonexistent():
    """is_loaded returns False for an unregistered model."""
    manager = ModelManager(budget_gb=8.0)
    loaded = await manager.is_loaded("nonexistent-model")
    assert loaded is False


# ---------------------------------------------------------------------------
# Multiple subclasses coexist
# ---------------------------------------------------------------------------


def test_multiple_subclass_types_coexist():
    """Different subclass types can all be instantiated together."""
    llm = MockLLM()
    embedder = MockEmbedder()
    transcriber = MockTranscriber()
    tts = MockTTS()
    image = MockImageProvider()
    video = MockVideoProvider()
    music = MockMusicProvider()
    voice = MockVoiceProvider()
    three_d = MockThreeDProvider()
    bg_rm = MockBgRemovalProvider()
    backend = MockMemoryBackend()

    assert llm.model_id == "mock-llm"
    assert embedder.model_id == "mock-embedder"
    assert transcriber.provider_id == "mock-stt"
    assert tts.provider_id == "mock-tts"
    assert image.provider_id == "mock-image"
    assert video.provider_id == "mock-video"
    assert music.provider_id == "mock-music"
    assert voice.provider_id == "mock-voice"
    assert three_d.provider_id == "mock-3d"
    assert bg_rm.provider_id == "mock-bgrm"
    assert isinstance(backend, MemoryBackend)


def test_subclass_with_extra_attributes():
    """Python subclasses can carry extra Python-only attributes."""

    class EnrichedLLM(CompletionModel):
        def __init__(self, nickname: str):
            super().__init__(model_id="enriched")
            self.nickname = nickname
            self.call_count = 0

    model = EnrichedLLM("speedy")
    assert model.model_id == "enriched"
    assert model.nickname == "speedy"
    assert model.call_count == 0
    model.call_count += 1
    assert model.call_count == 1
