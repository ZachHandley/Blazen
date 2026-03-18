/**
 * Multi-Step RAG Pipeline with Context Sharing
 *
 * Demonstrates how Blazen's workflow context lets steps share data across
 * a Retrieval-Augmented Generation (RAG) pipeline without passing large
 * payloads through events.
 *
 * Pipeline:
 *   StartEvent -> [parse_query] -> QueryEvent
 *                                    -> [retrieve] -> RetrievalEvent
 *                                                      -> [generate] -> StopEvent
 *
 * Context is used to carry the retrieved documents from the "retrieve"
 * step to the "generate" step. Both ctx.set() and ctx.get() are async
 * and accept any JSON-serializable value.
 *
 * Run with: node llm_rag_workflow.mjs
 */

import { Workflow } from "blazen";

const wf = new Workflow("rag-pipeline");

// ---------------------------------------------------------------------------
// Step 1 — Parse the user query
//
// Accepts the initial StartEvent, normalizes the raw query string, and
// emits a QueryEvent so downstream steps can act on the parsed intent.
// ---------------------------------------------------------------------------
wf.addStep("parse_query", ["blazen::StartEvent"], async (event, ctx) => {
  const rawQuery = event.data?.query ?? "What are the benefits of RAG pipelines?";

  console.log(`[parse_query] Received raw query: "${rawQuery}"`);

  // Simulate simple query parsing / intent extraction
  const parsed = {
    text: rawQuery.trim().toLowerCase(),
    keywords: rawQuery
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 3),
  };

  console.log(`[parse_query] Extracted keywords: ${parsed.keywords.join(", ")}`);

  // Store the parsed query in context so any later step can reference it
  await ctx.set("parsed_query", parsed);

  return { type: "QueryEvent", data: { keywords: parsed.keywords } };
});

// ---------------------------------------------------------------------------
// Step 2 — Retrieve relevant documents
//
// Uses the keywords from QueryEvent to simulate a vector-store lookup.
// The retrieved documents are stored in workflow context via ctx.set()
// so the generation step can read them without inflating the event payload.
// ---------------------------------------------------------------------------
wf.addStep("retrieve", ["QueryEvent"], async (event, ctx) => {
  const { keywords } = event.data;

  console.log(`[retrieve] Searching document store for: ${keywords.join(", ")}`);

  // Simulated document corpus — in a real system this would be a vector DB query
  const corpus = [
    {
      id: "doc-1",
      title: "Introduction to RAG",
      content:
        "Retrieval-Augmented Generation combines a retriever that fetches " +
        "relevant documents with a generator that produces answers grounded " +
        "in those documents, reducing hallucination.",
    },
    {
      id: "doc-2",
      title: "Benefits of RAG Pipelines",
      content:
        "RAG pipelines allow LLMs to access up-to-date knowledge without " +
        "retraining. They improve factual accuracy and let you control the " +
        "knowledge base independently of the model.",
    },
    {
      id: "doc-3",
      title: "Vector Databases Overview",
      content:
        "Vector databases store embeddings and support approximate nearest " +
        "neighbor search, making them the backbone of modern retrieval systems.",
    },
  ];

  // Simple keyword-matching retrieval (a real system would use embeddings)
  const retrieved = corpus.filter((doc) =>
    keywords.some(
      (kw) =>
        doc.title.toLowerCase().includes(kw) ||
        doc.content.toLowerCase().includes(kw)
    )
  );

  console.log(`[retrieve] Found ${retrieved.length} relevant document(s)`);

  // Share the retrieved documents with downstream steps through context.
  // This avoids bloating event payloads with large document bodies.
  await ctx.set("retrieved_docs", retrieved);

  return {
    type: "RetrievalEvent",
    data: { doc_count: retrieved.length },
  };
});

// ---------------------------------------------------------------------------
// Step 3 — Generate the final response
//
// Reads the retrieved documents from context via ctx.get(), combines them
// with the original parsed query, and produces a synthesized answer.
// ---------------------------------------------------------------------------
wf.addStep("generate", ["RetrievalEvent"], async (event, ctx) => {
  // Pull shared state out of context — no need to pass it through events
  const docs = await ctx.get("retrieved_docs");
  const query = await ctx.get("parsed_query");

  console.log(
    `[generate] Generating answer for "${query.text}" using ${docs.length} document(s)`
  );

  // Simulate LLM generation by combining document snippets into a response
  const combinedContext = docs.map((d) => d.content).join(" ");

  const response = [
    `Based on ${docs.length} retrieved document(s):`,
    "",
    ...docs.map((d) => `  - "${d.title}" (${d.id})`),
    "",
    "Generated answer:",
    `  ${combinedContext.slice(0, 200)}...`,
  ].join("\n");

  console.log(`[generate] Response ready (${response.length} chars)`);

  return {
    type: "blazen::StopEvent",
    result: {
      query: query.text,
      sources: docs.map((d) => d.id),
      response,
    },
  };
});

// ---------------------------------------------------------------------------
// Run the pipeline
// ---------------------------------------------------------------------------
console.log("=== RAG Pipeline Demo ===\n");

const result = await wf.run({ query: "What are the benefits of RAG pipelines?" });

console.log("\n=== Final Result ===");
console.log(`Query:    ${result.data.query}`);
console.log(`Sources:  ${result.data.sources.join(", ")}`);
console.log(`\n${result.data.response}`);
