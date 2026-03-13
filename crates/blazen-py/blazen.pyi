"""Type stubs for the blazen native module.

Provides IDE support (auto-completion, type checking) for the Blazen
Python bindings.
"""

from typing import Any, AsyncIterator, Optional, Union


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
    publishing.  All values must be JSON-serializable.
    """

    async def set(self, key: str, value: Any) -> None:
        """Store a JSON-serializable value under *key*."""
        ...

    async def get(self, key: str) -> Optional[Any]:
        """Retrieve a value previously stored under *key*."""
        ...

    async def send_event(self, event: Event) -> None:
        """Emit an event into the internal routing queue."""
        ...

    async def write_event_to_stream(self, event: Event) -> None:
        """Publish an event to the external broadcast stream."""
        ...

    async def run_id(self) -> str:
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
    """Decorator that wraps a Python async function as a workflow step.

    The decorated function must have the signature::

        async def my_step(ctx: Context, ev: Event) -> Event | list[Event] | None

    By default the step accepts ``StartEvent``.

    Can be used with or without arguments::

        @step
        async def my_step(ctx, ev): ...

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


class ChatMessage:
    """A single message in a chat conversation.

    Example::

        msg = ChatMessage("user", "Hello!")
        msg = ChatMessage.system("You are helpful.")
    """

    role: str
    content: Optional[str]

    def __init__(self, role: str, content: str) -> None: ...

    @staticmethod
    def system(content: str) -> "ChatMessage": ...
    @staticmethod
    def user(content: str) -> "ChatMessage": ...
    @staticmethod
    def assistant(content: str) -> "ChatMessage": ...
    @staticmethod
    def tool(content: str) -> "ChatMessage": ...


class CompletionModel:
    """A chat completion model with provider constructors.

    Example::

        model = CompletionModel.openai("sk-...")
        response = await model.complete([ChatMessage.user("Hi!")])
        print(response["content"])
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
    ) -> dict[str, Any]:
        """Perform a chat completion.

        Returns a dict with keys: ``content``, ``model``, ``tool_calls``,
        ``usage``, ``finish_reason``.
        """
        ...


__version__: str
