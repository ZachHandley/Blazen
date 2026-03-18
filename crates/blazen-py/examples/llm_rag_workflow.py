"""
Multi-step RAG (Retrieval-Augmented Generation) pipeline with context sharing.

Demonstrates how Blazen's Context object lets steps share data across a
workflow run.  Three steps form a classic RAG pipeline:

  1. parse_query   -- extracts a structured query from raw user input
  2. retrieve      -- fetches relevant documents and stores them in context
  3. generate      -- reads the documents from context and produces a response

Key points:
  - ctx.set(key, value) and ctx.get(key) are SYNCHRONOUS -- no await needed.
  - Any JSON-serializable value (dicts, lists, strings, numbers) can be stored.
  - Context persists across all steps within a single workflow run, so later
    steps can read whatever earlier steps wrote.

Run with: python llm_rag_workflow.py
"""

import asyncio

from blazen import Context, Event, StartEvent, StopEvent, Workflow, step


# ---------------------------------------------------------------------------
# Simulated knowledge base (in a real system this would be a vector store,
# database, or search index).
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE: list[dict[str, str]] = [
    {
        "id": "doc-1",
        "title": "Blazen Overview",
        "content": (
            "Blazen is a high-performance workflow engine written in Rust "
            "with Python bindings.  It uses an event-driven architecture "
            "where steps communicate through typed events."
        ),
    },
    {
        "id": "doc-2",
        "title": "Context Sharing",
        "content": (
            "The Context object provides synchronous get/set methods for "
            "sharing JSON-serializable data between steps within a single "
            "workflow run."
        ),
    },
    {
        "id": "doc-3",
        "title": "Event Routing",
        "content": (
            "Steps declare which event types they accept.  The workflow "
            "engine routes emitted events to the correct downstream step "
            "automatically."
        ),
    },
]


# ---------------------------------------------------------------------------
# Step 1: Parse the user query
# ---------------------------------------------------------------------------
@step  # Accepts StartEvent by default
async def parse_query(ctx: Context, ev: Event) -> Event:
    """Extract and normalise the raw user query.

    In a real pipeline you might call an LLM here to rephrase the query,
    extract keywords, or expand abbreviations.
    """
    raw_query: str = ev.query
    print(f"[parse_query] Received raw query: {raw_query!r}")

    # Simulate query processing -- lowercase and strip whitespace.
    processed_query = raw_query.strip().lower()

    # Store the processed query in context so later steps can reference it.
    ctx.set("original_query", raw_query)
    ctx.set("processed_query", processed_query)

    print(f"[parse_query] Processed query stored in context: {processed_query!r}")
    return Event("QueryEvent", query=processed_query)


# ---------------------------------------------------------------------------
# Step 2: Retrieve relevant documents
# ---------------------------------------------------------------------------
@step(accepts=["QueryEvent"])
async def retrieve(ctx: Context, ev: Event) -> Event:
    """Search the knowledge base for documents matching the query.

    A real implementation would call a vector store or embedding search.
    Here we do a simple keyword match for demonstration purposes.
    """
    query: str = ev.query
    print(f"[retrieve] Searching for documents matching: {query!r}")

    # Simple keyword matching against document content.
    matched_docs: list[dict[str, str]] = [
        doc
        for doc in KNOWLEDGE_BASE
        if any(word in doc["content"].lower() for word in query.split())
    ]

    if not matched_docs:
        # Fall back to returning all documents if nothing matched.
        matched_docs = KNOWLEDGE_BASE

    print(f"[retrieve] Found {len(matched_docs)} relevant document(s)")
    for doc in matched_docs:
        print(f"  - {doc['title']} ({doc['id']})")

    # Store retrieved documents in context for the generate step.
    ctx.set("retrieved_docs", matched_docs)
    ctx.set("num_docs_retrieved", len(matched_docs))

    return Event("RetrievalEvent", num_results=len(matched_docs))


# ---------------------------------------------------------------------------
# Step 3: Generate a response
# ---------------------------------------------------------------------------
@step(accepts=["RetrievalEvent"])
async def generate(ctx: Context, ev: Event) -> StopEvent:
    """Compose a final answer from the retrieved documents.

    In production this step would send the query + retrieved context to an
    LLM.  Here we simulate the response by concatenating document snippets.
    """
    # Read shared data from context -- no await needed.
    original_query: str = ctx.get("original_query")
    retrieved_docs: list[dict[str, str]] = ctx.get("retrieved_docs")
    num_docs: int = ctx.get("num_docs_retrieved")

    print(f"[generate] Building response from {num_docs} document(s)")

    # Simulate LLM generation by summarising the retrieved content.
    context_block = "\n\n".join(
        f"[{doc['title']}]: {doc['content']}" for doc in retrieved_docs
    )
    simulated_response = (
        f"Based on {num_docs} retrieved document(s), here is the answer to "
        f"your question \"{original_query}\":\n\n{context_block}"
    )

    print("[generate] Response generated successfully")

    return StopEvent(result={"response": simulated_response, "sources": num_docs})


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def main() -> None:
    # Build the workflow from the three RAG steps.
    wf = Workflow("rag-pipeline", [parse_query, retrieve, generate])

    # Run the workflow with a sample query passed as the StartEvent payload.
    handler = await wf.run(query="How does Blazen share context between steps?")

    # Await the final result (a StopEvent).
    result = await handler.result()
    output = result.to_dict()

    print("\n" + "=" * 60)
    print("FINAL RESULT")
    print("=" * 60)
    print(f"Sources consulted: {output['result']['sources']}")
    print(f"\n{output['result']['response']}")


if __name__ == "__main__":
    asyncio.run(main())
