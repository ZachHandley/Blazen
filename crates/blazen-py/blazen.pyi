"""Type stubs for the blazen native module.

Provides IDE support (auto-completion, type checking) for the Blazen
Python bindings.
"""

from typing import Any, AsyncIterator, Callable, Coroutine, Optional, Union


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
    publishing.  All values must be JSON-serializable (for ``set``/``get``)
    or raw bytes (for ``set_bytes``/``get_bytes``).

    All context methods are synchronous (they block on in-memory
    operations), so they can be called from both sync and async steps
    without ``await``.
    """

    def set(self, key: str, value: Any) -> None:
        """Store a JSON-serializable value under *key*."""
        ...

    def get(self, key: str) -> Optional[Any]:
        """Retrieve a value previously stored under *key*."""
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
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> CompletionResponse:
        """Perform a chat completion.

        Returns a CompletionResponse with attributes: ``content``, ``model``,
        ``tool_calls``, ``usage``, ``finish_reason``.
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


__version__: str
