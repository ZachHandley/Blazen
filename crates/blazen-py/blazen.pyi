"""Type stubs for the blazen native module.

Provides IDE support (auto-completion, type checking) for the Blazen
Python bindings.
"""

from typing import Any, AsyncIterator, Callable, ClassVar, Coroutine, Optional, Protocol, Union

StateValue = Any
"""Any Python value that can be stored in the workflow context.

JSON-serializable types (dict, list, str, int, float, bool, None) are stored
efficiently as JSON. ``bytes``/``bytearray`` are stored as raw binary. All
other objects (Pydantic models, custom classes, etc.) are pickled automatically
and deserialized back to their original type on retrieval."""


class FieldStore(Protocol):
    """Custom persistence strategy for a BlazenState field.

    Implement this protocol to provide custom save/load behavior
    for specific fields (e.g., S3 storage, database, etc.).
    """

    def save(self, key: str, value: Any, ctx: "Context") -> None:
        """Persist the field value."""
        ...

    def load(self, key: str, ctx: "Context") -> Any:
        """Load the field value."""
        ...


class CallbackFieldStore:
    """Store/load a field via user-provided callables.

    Example::

        CallbackFieldStore(
            save_fn=lambda k, v: s3.put_object(Bucket="b", Key=k, Body=v),
            load_fn=lambda k: s3.get_object(Bucket="b", Key=k)["Body"].read(),
        )
    """

    def __init__(
        self,
        save_fn: Callable[[str, Any], None],
        load_fn: Callable[[str], Any],
    ) -> None: ...
    def save(self, key: str, value: Any, ctx: "Context") -> None: ...
    def load(self, key: str, ctx: "Context") -> Any: ...


class BlazenState:
    """Base class for typed workflow state with per-field context storage.

    Subclass with ``@dataclass`` to get typed, per-field storage where each
    field is stored individually with its optimal tier. Fields in
    ``Meta.transient`` are excluded from serialization and recreated via
    ``restore()``.

    Example::

        @dataclass
        class MyState(BlazenState):
            input_path: str = ""
            conn: sqlite3.Connection | None = None

            class Meta:
                transient = {"conn"}
                store_by = {}

            def restore(self):
                if self.input_path:
                    self.conn = sqlite3.connect(self.input_path)
    """

    class Meta:
        """Configuration for field storage behavior."""

        transient: ClassVar[set[str]]
        """Field names excluded from serialization."""
        store_by: ClassVar[dict[str, FieldStore]]
        """Custom persistence strategies per field."""

    def restore(self) -> None:
        """Override to recreate transient fields after deserialization.

        Called automatically when the state is restored from a snapshot
        or context. All serializable fields are already set; transient
        fields are ``None``.
        """
        ...


# ---------------------------------------------------------------------------
# Error types
# ---------------------------------------------------------------------------


class BlazenError(Exception):
    """Base exception for all Blazen errors."""

    ...


class AuthError(BlazenError):
    """Authentication or authorization failure."""

    ...


class RateLimitError(BlazenError):
    """Rate limit exceeded."""

    ...


class TimeoutError(BlazenError):
    """Operation timed out."""

    ...


class ValidationError(BlazenError):
    """Input validation failure."""

    ...


class ContentPolicyError(BlazenError):
    """Content policy violation."""

    ...


class ProviderError(BlazenError):
    """Provider-level error (HTTP errors, network failures, etc.)."""

    ...


class UnsupportedError(BlazenError):
    """The requested operation is not supported by this provider."""

    ...


class ComputeError(BlazenError):
    """Compute job error."""

    ...


class MediaError(BlazenError):
    """Media processing error."""

    ...


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class Event:
    """A dict-like event object for inter-step communication.

    Attributes can be accessed directly (e.g. ``ev.text``) and are
    backed by a JSON data store.

    Example::

        ev = Event("AnalyzeEvent", text="hello", score=0.9)
        print(ev.text)       # "hello"
        print(ev.to_dict())  # {"text": "hello", "score": 0.9}
    """

    event_type: str

    def __init__(self, event_type: str, **kwargs: Any) -> None: ...
    def __getattr__(self, name: str) -> Any: ...
    def to_dict(self) -> dict[str, Any]: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...


class StartEvent(Event):
    """Event that kicks off a workflow.

    Example::

        ev = StartEvent(text="hello", count=5)
    """

    def __init__(self, **kwargs: Any) -> None: ...


class StopEvent(Event):
    """Event that terminates a workflow with a result.

    Example::

        ev = StopEvent(result={"answer": 42})
    """

    def __init__(self, **kwargs: Any) -> None: ...


class Context:
    """Shared workflow context accessible by all steps.

    Provides typed key/value storage, event emission, and stream
    publishing.  Accepts any Python value — JSON-serializable types are
    stored as JSON, ``bytes``/``bytearray`` as raw binary, and all other
    objects (Pydantic models, custom classes, etc.) are pickled automatically.

    All context methods are synchronous (they block on in-memory
    operations), so they can be called from both sync and async steps
    without ``await``.
    """

    def set(self, key: str, value: StateValue) -> None:
        """Store any value under *key*.

        - ``bytes``/``bytearray`` → stored as raw binary (survives snapshots)
        - JSON-serializable (dict, list, str, int, float, bool, None) → stored as JSON (survives snapshots)
        - Picklable objects (Pydantic, dataclasses) → pickled automatically (survives snapshots)
        - Unpicklable objects (DB connections, file handles) → stored as live reference (same-process only)
        """
        ...

    def get(self, key: str) -> Optional[StateValue]:
        """Retrieve the value stored under *key*, deserialized to its original type.

        Returns the original Python type for JSON values, ``bytes`` for
        binary data, the original object for pickled or live-reference
        values, or ``None`` if the key does not exist.
        """
        ...

    def set_bytes(self, key: str, data: bytes) -> None:
        """Store raw binary data under *key*.

        Useful for files, images, serialized objects, or any data that
        should not be JSON-serialized. Persists through pause/resume.
        """
        ...

    def get_bytes(self, key: str) -> Optional[bytes]:
        """Retrieve raw binary data previously stored under *key*.

        Returns ``None`` if the key does not exist or holds JSON data.
        """
        ...

    def send_event(self, event: Event) -> None:
        """Emit an event into the internal routing queue."""
        ...

    def write_event_to_stream(self, event: Event) -> None:
        """Publish an event to the external broadcast stream."""
        ...

    def run_id(self) -> str:
        """Get the workflow run ID as a UUID string."""
        ...


class _StepWrapper:
    """Internal wrapper created by the ``@step`` decorator.

    You should not instantiate this directly; use ``@step`` instead.
    """

    name: str
    accepts: list[str]
    emits: list[str]
    max_concurrency: int

    async def __call__(self, ctx: Context, event: Event) -> Event: ...


def step(
    func: Any = None,
    *,
    accepts: Optional[list[str]] = None,
    emits: Optional[list[str]] = None,
    max_concurrency: int = 0,
) -> Any:
    """Decorator that wraps a Python function as a workflow step.

    The decorated function can be either sync or async and must have
    the signature::

        def my_step(ctx: Context, ev: Event) -> Event | list[Event] | None
        async def my_step(ctx: Context, ev: Event) -> Event | list[Event] | None

    By default the step accepts ``StartEvent``.

    Can be used with or without arguments::

        @step
        def my_step(ctx, ev): ...

        @step(accepts=["MyEvent"], emits=["ResultEvent"])
        async def my_step(ctx, ev): ...
    """
    ...


class WorkflowHandler:
    """Handle to a running workflow.

    Example::

        handler = await wf.run(prompt="Hello")
        result = await handler.result()
        print(result.to_dict())
    """

    async def result(self) -> Event:
        """Await the final workflow result (consumes the handler)."""
        ...

    def stream_events(self) -> "_EventStream":
        """Create an async iterator over intermediate events."""
        ...


class _EventStream:
    """Async iterator over streamed workflow events.

    Use with ``async for``::

        async for event in handler.stream_events():
            print(event.event_type, event.to_dict())
    """

    def __aiter__(self) -> "_EventStream": ...
    async def __anext__(self) -> Event: ...


class Workflow:
    """A validated, ready-to-run workflow.

    Example::

        @step
        async def echo(ctx, ev):
            return StopEvent(result=ev.to_dict())

        wf = Workflow("echo-wf", [echo])
        handler = await wf.run(message="hello")
    """

    def __init__(
        self,
        name: str,
        steps: list[_StepWrapper],
        timeout: Optional[float] = None,
    ) -> None: ...

    async def run(self, **kwargs: Any) -> WorkflowHandler:
        """Execute the workflow with keyword arguments as the start payload."""
        ...


# ---------------------------------------------------------------------------
# LLM types
# ---------------------------------------------------------------------------


class Role:
    """Role constants for chat messages.

    Example::

        ChatMessage(role=Role.USER, content="Hello!")
        ChatMessage(role=Role.SYSTEM, content="You are helpful.")
    """

    SYSTEM: str
    USER: str
    ASSISTANT: str
    TOOL: str


class ContentPart:
    """A single part in a multimodal message.

    Use the static factory methods to create parts::

        ContentPart.text(text="Hello")
        ContentPart.image_url(url="https://...", media_type="image/png")
        ContentPart.image_base64(data="...", media_type="image/jpeg")
    """

    @staticmethod
    def text(*, text: str) -> "ContentPart":
        """Create a text content part."""
        ...

    @staticmethod
    def image_url(*, url: str, media_type: Optional[str] = None) -> "ContentPart":
        """Create an image content part from a URL."""
        ...

    @staticmethod
    def image_base64(*, data: str, media_type: str) -> "ContentPart":
        """Create an image content part from base64 data."""
        ...


class ChatMessage:
    """A single message in a chat conversation.

    Example::

        msg = ChatMessage(role=Role.USER, content="Hello!")
        msg = ChatMessage(content="Hello!")  # defaults to role="user"
        msg = ChatMessage.system("You are helpful.")

        # Multimodal
        msg = ChatMessage(role=Role.USER, parts=[
            ContentPart.text(text="Describe this"),
            ContentPart.image_url(url="https://..."),
        ])
    """

    role: str
    content: Optional[str]

    def __init__(
        self,
        role: str = "user",
        content: Optional[str] = None,
        parts: Optional[list[ContentPart]] = None,
    ) -> None: ...

    @staticmethod
    def system(content: str) -> "ChatMessage": ...
    @staticmethod
    def user(content: str) -> "ChatMessage": ...
    @staticmethod
    def assistant(content: str) -> "ChatMessage": ...
    @staticmethod
    def tool(content: str) -> "ChatMessage": ...

    @staticmethod
    def user_image_url(
        *, text: str, url: str, media_type: Optional[str] = None
    ) -> "ChatMessage":
        """Create a user message with text and an image URL."""
        ...

    @staticmethod
    def user_image_base64(
        *, text: str, data: str, media_type: str
    ) -> "ChatMessage":
        """Create a user message with text and a base64-encoded image."""
        ...

    @staticmethod
    def user_parts(*, parts: list[ContentPart]) -> "ChatMessage":
        """Create a user message from a list of ContentPart objects."""
        ...


class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]

    def __getitem__(self, key: str) -> Any: ...


class TokenUsage:
    """Token usage statistics for a completion."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __getitem__(self, key: str) -> Any: ...


class RequestTiming:
    """Timing metadata for a request."""

    queue_ms: Optional[int]
    execution_ms: Optional[int]
    total_ms: Optional[int]

    def __getitem__(self, key: str) -> Any: ...


class CompletionResponse:
    """The result of a chat completion.

    Supports both attribute access and dict-style access::

        response.content        # attribute
        response["content"]     # dict-style (backwards compat)
    """

    content: Optional[str]
    model: str
    finish_reason: Optional[str]
    tool_calls: list[ToolCall]
    usage: Optional[TokenUsage]
    cost: Optional[float]
    timing: Optional[RequestTiming]
    images: list[dict[str, Any]]
    audio: list[dict[str, Any]]
    videos: list[dict[str, Any]]

    def __getitem__(self, key: str) -> Any: ...
    def keys(self) -> list[str]: ...


class CompletionOptions:
    """Options for a chat completion request.

    All fields are optional. Pass an instance to ``CompletionModel.complete()``
    or ``CompletionModel.stream()`` to customise the request::

        opts = CompletionOptions(temperature=0.7, max_tokens=1024)
        response = await model.complete(messages, opts)
    """

    temperature: Optional[float]
    """Sampling temperature (0.0-2.0)."""
    max_tokens: Optional[int]
    """Maximum tokens to generate."""
    top_p: Optional[float]
    """Nucleus sampling parameter (0.0-1.0)."""
    model: Optional[str]
    """Model override for this request."""
    tools: Optional[Any]
    """Tool definitions for function calling. Each tool is a dict with
    ``name``, ``description``, and ``parameters`` keys."""
    response_format: Optional[dict[str, Any]]
    """JSON schema dict for structured output."""

    def __init__(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        model: Optional[str] = None,
        tools: Optional[Any] = None,
        response_format: Optional[dict[str, Any]] = None,
    ) -> None: ...


class CompletionModel:
    """A chat completion model with provider constructors.

    Example::

        model = CompletionModel.openai("sk-...")
        response = await model.complete([ChatMessage.user("Hi!")])
        print(response.content)
    """

    model_id: str

    @staticmethod
    def openai(api_key: str, model: Optional[str] = None) -> "CompletionModel": ...
    @staticmethod
    def anthropic(api_key: str, model: Optional[str] = None) -> "CompletionModel": ...
    @staticmethod
    def gemini(api_key: str, model: Optional[str] = None) -> "CompletionModel": ...
    @staticmethod
    def azure(
        api_key: str, resource_name: str, deployment_name: str
    ) -> "CompletionModel": ...
    @staticmethod
    def openrouter(api_key: str, model: Optional[str] = None) -> "CompletionModel": ...
    @staticmethod
    def groq(api_key: str, model: Optional[str] = None) -> "CompletionModel": ...
    @staticmethod
    def together(api_key: str, model: Optional[str] = None) -> "CompletionModel": ...
    @staticmethod
    def mistral(api_key: str, model: Optional[str] = None) -> "CompletionModel": ...
    @staticmethod
    def deepseek(api_key: str, model: Optional[str] = None) -> "CompletionModel": ...
    @staticmethod
    def fireworks(api_key: str, model: Optional[str] = None) -> "CompletionModel": ...
    @staticmethod
    def perplexity(api_key: str, model: Optional[str] = None) -> "CompletionModel": ...
    @staticmethod
    def xai(api_key: str, model: Optional[str] = None) -> "CompletionModel": ...
    @staticmethod
    def cohere(api_key: str, model: Optional[str] = None) -> "CompletionModel": ...
    @staticmethod
    def bedrock(
        api_key: str, region: str, model: Optional[str] = None
    ) -> "CompletionModel": ...
    @staticmethod
    def fal(api_key: str, model: Optional[str] = None) -> "CompletionModel": ...

    async def complete(
        self,
        messages: list[ChatMessage],
        options: Optional[CompletionOptions] = None,
    ) -> CompletionResponse:
        """Perform a chat completion.

        Args:
            messages: A list of ChatMessage objects.
            options: Optional CompletionOptions to customise the request.

        Returns a CompletionResponse with attributes: ``content``, ``model``,
        ``tool_calls``, ``usage``, ``finish_reason``.
        """
        ...

    async def stream(
        self,
        messages: list[ChatMessage],
        on_chunk: Callable[[dict[str, Any]], Any],
        options: Optional[CompletionOptions] = None,
    ) -> None:
        """Stream a chat completion, calling ``on_chunk`` for each chunk.

        Each chunk is a dict with keys: ``delta`` (optional str),
        ``finish_reason`` (optional str), ``tool_calls`` (list of dicts).

        Example::

            def handle(chunk):
                if chunk["delta"]:
                    print(chunk["delta"], end="")

            await model.stream([ChatMessage.user("Hi!")], handle)
        """
        ...


# ---------------------------------------------------------------------------
# Agent types
# ---------------------------------------------------------------------------


class ToolDef:
    """A tool definition for the agent.

    The handler can be either a sync function or an async function.

    Example::

        # Sync handler
        tool = ToolDef(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=lambda args: {"results": []},
        )

        # Async handler
        async def async_search(args):
            result = await some_async_api(args["query"])
            return {"results": result}

        tool = ToolDef(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            handler=async_search,
        )
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Union[
            Callable[[dict[str, Any]], Any],
            Callable[[dict[str, Any]], Coroutine[Any, Any, Any]],
        ],
    ) -> None: ...


class AgentResult:
    """Result of an agent run.

    Example::

        result = await run_agent(model, messages, tools=[tool])
        print(result.response.content)
        print(result.iterations)
    """

    response: CompletionResponse
    messages: list[ChatMessage]
    iterations: int
    total_cost: Optional[float]


async def run_agent(
    model: CompletionModel,
    messages: list[ChatMessage],
    *,
    tools: list[ToolDef],
    max_iterations: int = 10,
    system_prompt: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    add_finish_tool: bool = False,
) -> AgentResult:
    """Run an agentic tool execution loop.

    Sends messages to the model with tool definitions, executes tool calls,
    feeds results back, and repeats until the model stops calling tools
    or ``max_iterations`` is reached.
    """
    ...


# ---------------------------------------------------------------------------
# Compute / Media types
# ---------------------------------------------------------------------------


class MediaType:
    """Media type constants (MIME strings) for identifying file formats.

    Example::

        MediaType.PNG   # "image/png"
        MediaType.MP4   # "video/mp4"
        MediaType.MP3   # "audio/mpeg"
    """

    # Images
    PNG: str
    JPEG: str
    WEBP: str
    GIF: str
    SVG: str
    BMP: str
    TIFF: str
    AVIF: str

    # Video
    MP4: str
    WEBM: str
    MOV: str

    # Audio
    MP3: str
    WAV: str
    OGG: str
    FLAC: str
    AAC: str
    M4A: str

    # 3D Models
    GLB: str
    GLTF: str
    OBJ: str
    USDZ: str
    FBX: str
    STL: str

    # Documents
    PDF: str


class ImageRequest:
    """Request to generate images from a text prompt.

    Example::

        req = ImageRequest(prompt="a cat in space", width=1024, height=1024)
    """

    prompt: str
    negative_prompt: Optional[str]
    width: Optional[int]
    height: Optional[int]
    num_images: Optional[int]
    model: Optional[str]

    def __init__(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_images: Optional[int] = None,
        model: Optional[str] = None,
    ) -> None: ...


class UpscaleRequest:
    """Request to upscale an image.

    Example::

        req = UpscaleRequest(image_url="https://...", scale=4.0)
    """

    image_url: str
    scale: float
    model: Optional[str]

    def __init__(
        self,
        *,
        image_url: str,
        scale: float,
        model: Optional[str] = None,
    ) -> None: ...


class VideoRequest:
    """Request to generate a video.

    Example::

        req = VideoRequest(prompt="a sunset timelapse", duration_seconds=5.0)
        req = VideoRequest(prompt="animate this", image_url="https://...")
    """

    prompt: str
    image_url: Optional[str]
    duration_seconds: Optional[float]
    negative_prompt: Optional[str]
    width: Optional[int]
    height: Optional[int]
    model: Optional[str]

    def __init__(
        self,
        *,
        prompt: str,
        image_url: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        model: Optional[str] = None,
    ) -> None: ...


class SpeechRequest:
    """Request to generate speech from text.

    Example::

        req = SpeechRequest(text="Hello world", voice="alloy", speed=1.2)
    """

    text: str
    voice: Optional[str]
    voice_url: Optional[str]
    language: Optional[str]
    speed: Optional[float]
    model: Optional[str]

    def __init__(
        self,
        *,
        text: str,
        voice: Optional[str] = None,
        voice_url: Optional[str] = None,
        language: Optional[str] = None,
        speed: Optional[float] = None,
        model: Optional[str] = None,
    ) -> None: ...


class MusicRequest:
    """Request to generate music or sound effects.

    Example::

        req = MusicRequest(prompt="upbeat jazz", duration_seconds=30.0)
    """

    prompt: str
    duration_seconds: Optional[float]
    model: Optional[str]

    def __init__(
        self,
        *,
        prompt: str,
        duration_seconds: Optional[float] = None,
        model: Optional[str] = None,
    ) -> None: ...


class TranscriptionRequest:
    """Request to transcribe audio to text.

    Example::

        req = TranscriptionRequest(audio_url="https://...", language="en", diarize=True)
    """

    audio_url: str
    language: Optional[str]
    diarize: bool
    model: Optional[str]

    def __init__(
        self,
        *,
        audio_url: str,
        language: Optional[str] = None,
        diarize: Optional[bool] = None,
        model: Optional[str] = None,
    ) -> None: ...


class ThreeDRequest:
    """Request to generate a 3D model.

    Example::

        req = ThreeDRequest(prompt="a 3D cat", format="glb")
        req = ThreeDRequest(image_url="https://...", format="obj")
    """

    prompt: str
    image_url: Optional[str]
    format: Optional[str]
    model: Optional[str]

    def __init__(
        self,
        *,
        prompt: Optional[str] = None,
        image_url: Optional[str] = None,
        format: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None: ...


# ---------------------------------------------------------------------------
# Job types
# ---------------------------------------------------------------------------


class JobStatus:
    """Job status constants.

    Example::

        JobStatus.QUEUED     # "queued"
        JobStatus.RUNNING    # "running"
        JobStatus.COMPLETED  # "completed"
    """

    QUEUED: str
    RUNNING: str
    COMPLETED: str
    FAILED: str
    CANCELLED: str


class JobHandle:
    """A handle to a submitted compute job.

    Example::

        handle = await fal.submit(model="fal-ai/flux/dev", input={...})
        print(handle.id, handle.model)
    """

    id: str
    provider: str
    model: str
    submitted_at: str

    def __getitem__(self, key: str) -> Any: ...


class ComputeRequest:
    """Input for a raw compute job.

    Example::

        req = ComputeRequest(model="fal-ai/flux/dev", input={"prompt": "a cat"})
    """

    model: str
    input: dict[str, Any]

    def __init__(self, *, model: str, input: dict[str, Any]) -> None: ...


# ---------------------------------------------------------------------------
# Media output types
# ---------------------------------------------------------------------------


class MediaOutput:
    """A single piece of generated media content.

    At least one of url, base64, or raw_content will be populated.

    Example::

        output.url
        output.media_type
        output.file_size
    """

    url: Optional[str]
    base64: Optional[str]
    raw_content: Optional[str]
    media_type: str
    file_size: Optional[int]
    metadata: dict[str, Any]

    def __getitem__(self, key: str) -> Any: ...


class GeneratedImage:
    """A single generated image with optional dimension metadata.

    Example::

        img.media.url
        img.width, img.height
    """

    media: MediaOutput
    width: Optional[int]
    height: Optional[int]

    def __getitem__(self, key: str) -> Any: ...


class GeneratedVideo:
    """A single generated video with optional metadata.

    Example::

        vid.media.url
        vid.duration_seconds, vid.fps
    """

    media: MediaOutput
    width: Optional[int]
    height: Optional[int]
    duration_seconds: Optional[float]
    fps: Optional[float]

    def __getitem__(self, key: str) -> Any: ...


class GeneratedAudio:
    """A single generated audio clip with optional metadata.

    Example::

        audio.media.url
        audio.duration_seconds, audio.sample_rate
    """

    media: MediaOutput
    duration_seconds: Optional[float]
    sample_rate: Optional[int]
    channels: Optional[int]

    def __getitem__(self, key: str) -> Any: ...


class Generated3DModel:
    """A single generated 3D model with optional mesh metadata.

    Example::

        model.media.url
        model.vertex_count, model.has_textures
    """

    media: MediaOutput
    vertex_count: Optional[int]
    face_count: Optional[int]
    has_textures: bool
    has_animations: bool

    def __getitem__(self, key: str) -> Any: ...


# ---------------------------------------------------------------------------
# FalProvider
# ---------------------------------------------------------------------------


class FalProvider:
    """A fal.ai provider for image gen, video, audio, TTS, transcription, and LLM.

    This is the unified entry point for all fal.ai capabilities:
    image generation and upscaling, video generation (text-to-video,
    image-to-video), audio generation (TTS, music, sound effects),
    audio transcription, raw compute job submission, and LLM chat
    completions (via fal-ai/any-llm).

    Example::

        fal = FalProvider(api_key="fal-key-...")
        result = await fal.generate_image(ImageRequest(prompt="a cat in space"))
        response = await fal.complete([ChatMessage.user("Hello!")])
    """

    def __init__(self, *, api_key: str, model: Optional[str] = None) -> None: ...

    @property
    def model_id(self) -> str:
        """Get the model ID."""
        ...

    # -- Image -----------------------------------------------------------------

    async def generate_image(self, request: ImageRequest) -> dict[str, Any]:
        """Generate images from a text prompt.

        Args:
            request: An ImageRequest with prompt, dimensions, etc.

        Returns:
            A dict with images, timing, cost, and metadata.
        """
        ...

    async def upscale_image(self, request: UpscaleRequest) -> dict[str, Any]:
        """Upscale an image.

        Args:
            request: An UpscaleRequest with image_url and scale factor.

        Returns:
            A dict with the upscaled image, timing, cost, and metadata.
        """
        ...

    # -- Video -----------------------------------------------------------------

    async def text_to_video(self, request: VideoRequest) -> dict[str, Any]:
        """Generate a video from a text prompt.

        Args:
            request: A VideoRequest with prompt and optional parameters.

        Returns:
            A dict with videos, timing, cost, and metadata.
        """
        ...

    async def image_to_video(self, request: VideoRequest) -> dict[str, Any]:
        """Generate a video from a source image and prompt.

        Args:
            request: A VideoRequest with prompt and image_url.

        Returns:
            A dict with videos, timing, cost, and metadata.
        """
        ...

    # -- Audio -----------------------------------------------------------------

    async def text_to_speech(self, request: SpeechRequest) -> dict[str, Any]:
        """Synthesize speech from text.

        Args:
            request: A SpeechRequest with text and optional voice/language.

        Returns:
            A dict with audio clips, timing, cost, and metadata.
        """
        ...

    async def generate_music(self, request: MusicRequest) -> dict[str, Any]:
        """Generate music from a prompt.

        Args:
            request: A MusicRequest with prompt and optional duration.

        Returns:
            A dict with audio clips, timing, cost, and metadata.
        """
        ...

    async def generate_sfx(self, request: MusicRequest) -> dict[str, Any]:
        """Generate sound effects from a prompt.

        Args:
            request: A MusicRequest with prompt and optional duration.

        Returns:
            A dict with audio clips, timing, cost, and metadata.
        """
        ...

    # -- Transcription ---------------------------------------------------------

    async def transcribe(self, request: TranscriptionRequest) -> dict[str, Any]:
        """Transcribe audio to text.

        Args:
            request: A TranscriptionRequest with audio_url and options.

        Returns:
            A dict with text, segments, language, timing, cost, and metadata.
        """
        ...

    # -- Raw compute -----------------------------------------------------------

    async def run(self, *, model: str, input: dict[str, Any]) -> dict[str, Any]:
        """Submit a compute job and wait for the result.

        Args:
            model: The fal.ai model endpoint (e.g. "fal-ai/flux/dev").
            input: Input parameters as a dict.

        Returns:
            A dict with output, timing, cost, and metadata.
        """
        ...

    async def submit(self, *, model: str, input: dict[str, Any]) -> dict[str, Any]:
        """Submit a compute job without waiting (returns a job handle dict).

        Args:
            model: The fal.ai model endpoint.
            input: Input parameters as a dict.

        Returns:
            A dict with id, provider, model, and submitted_at.
        """
        ...

    async def status(self, *, job_id: str, model: str) -> str:
        """Poll the status of a submitted job.

        Args:
            job_id: The job identifier returned by submit().
            model: The model endpoint the job was submitted to.

        Returns:
            A status string: "queued", "running", "completed", "failed",
            or "cancelled".
        """
        ...

    async def cancel(self, *, job_id: str, model: str) -> None:
        """Cancel a running or queued job.

        Args:
            job_id: The job identifier returned by submit().
            model: The model endpoint the job was submitted to.
        """
        ...

    # -- LLM -------------------------------------------------------------------

    async def complete(
        self,
        messages: list[ChatMessage],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        response_format: Optional[dict[str, Any]] = None,
    ) -> CompletionResponse:
        """Perform a chat completion via fal-ai/any-llm.

        Args:
            messages: A list of ChatMessage objects.
            temperature: Optional sampling temperature (0.0-2.0).
            max_tokens: Optional maximum tokens to generate.
            model: Optional model override for this request.
            response_format: Optional JSON schema dict for structured output.

        Returns:
            A CompletionResponse with content, model, tool_calls, usage, etc.
        """
        ...

    async def stream(
        self,
        messages: list[ChatMessage],
        on_chunk: Callable[[dict[str, Any]], Any],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> None:
        """Stream a chat completion, calling a callback for each chunk.

        Args:
            messages: A list of ChatMessage objects.
            on_chunk: Callback function receiving each chunk as a dict.
            temperature: Optional sampling temperature (0.0-2.0).
            max_tokens: Optional maximum tokens to generate.
            model: Optional model override for this request.
        """
        ...


__version__: str
