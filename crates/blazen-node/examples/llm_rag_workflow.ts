/**
 * Multi-step RAG (Retrieval-Augmented Generation) pipeline with context sharing.
 *
 * Demonstrates how Blazen's Context object lets steps share data across a
 * workflow run.  Three steps form a classic RAG pipeline:
 *
 *   1. parse_query   -- extracts a structured query from raw user input
 *   2. retrieve      -- fetches relevant documents and stores them in context
 *   3. generate      -- reads the documents from context and produces a response
 *
 * Key points:
 *   - ctx.set(key, value) and ctx.get(key) are async -- use await.
 *   - Any JSON-serializable value (objects, arrays, strings, numbers) can be stored.
 *   - Context persists across all steps within a single workflow run, so later
 *     steps can read whatever earlier steps wrote.
 *
 * Run with: npx tsx llm_rag_workflow.ts
 */

import { Workflow } from "blazen";
import type { Context, JsWorkflowResult } from "blazen";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** A single document in the knowledge base. */
interface CorpusDocument {
  id: string;
  title: string;
  content: string;
}

/** The structured form of a user query after parsing. */
interface ParsedQuery {
  original: string;
  processed: string;
}

// ---------------------------------------------------------------------------
// Simulated knowledge base (in a real system this would be a vector store,
// database, or search index).
// ---------------------------------------------------------------------------
const KNOWLEDGE_BASE: CorpusDocument[] = [
  {
    id: "doc-1",
    title: "Blazen Overview",
    content:
      "Blazen is a high-performance workflow engine written in Rust " +
      "with Node.js bindings.  It uses an event-driven architecture " +
      "where steps communicate through typed events.",
  },
  {
    id: "doc-2",
    title: "Context Sharing",
    content:
      "The Context object provides async get/set methods for " +
      "sharing JSON-serializable data between steps within a single " +
      "workflow run.",
  },
  {
    id: "doc-3",
    title: "Event Routing",
    content:
      "Steps declare which event types they accept.  The workflow " +
      "engine routes emitted events to the correct downstream step " +
      "automatically.",
  },
];

// ---------------------------------------------------------------------------
// Build the workflow
// ---------------------------------------------------------------------------
const wf = new Workflow("rag-pipeline");

// ---------------------------------------------------------------------------
// Step 1: Parse the user query
// ---------------------------------------------------------------------------
wf.addStep(
  "parse_query",
  ["blazen::StartEvent"],
  async (event: Record<string, any>, ctx: Context) => {
    /**
     * Extract and normalise the raw user query.
     *
     * In a real pipeline you might call an LLM here to rephrase the query,
     * extract keywords, or expand abbreviations.
     */
    const rawQuery: string = event.query || "What is Blazen?";
    console.log(`[parse_query] Received raw query: "${rawQuery}"`);

    // Simulate query processing -- lowercase and strip whitespace.
    const processedQuery: string = rawQuery.trim().toLowerCase();

    // Store the processed query in context so later steps can reference it.
    await ctx.set("original_query", rawQuery);
    await ctx.set("processed_query", processedQuery);

    const parsed: ParsedQuery = {
      original: rawQuery,
      processed: processedQuery,
    };

    console.log(
      `[parse_query] Processed query stored in context: "${parsed.processed}"`,
    );

    return {
      type: "QueryEvent",
      query: processedQuery,
    };
  },
);

// ---------------------------------------------------------------------------
// Step 2: Retrieve relevant documents
// ---------------------------------------------------------------------------
wf.addStep(
  "retrieve",
  ["QueryEvent"],
  async (event: Record<string, any>, ctx: Context) => {
    /**
     * Search the knowledge base for documents matching the query.
     *
     * A real implementation would call a vector store or embedding search.
     * Here we do a simple keyword match for demonstration purposes.
     */
    const query: string = event.query;
    console.log(`[retrieve] Searching for documents matching: "${query}"`);

    // Simple keyword matching against document content.
    let matchedDocs: CorpusDocument[] = KNOWLEDGE_BASE.filter(
      (doc: CorpusDocument) =>
        query
          .split(" ")
          .some((word: string) => doc.content.toLowerCase().includes(word)),
    );

    if (matchedDocs.length === 0) {
      // Fall back to returning all documents if nothing matched.
      matchedDocs = KNOWLEDGE_BASE;
    }

    console.log(`[retrieve] Found ${matchedDocs.length} relevant document(s)`);
    for (const doc of matchedDocs) {
      console.log(`  - ${doc.title} (${doc.id})`);
    }

    // Store retrieved documents in context for the generate step.
    await ctx.set("retrieved_docs", matchedDocs);
    await ctx.set("num_docs_retrieved", matchedDocs.length);

    return {
      type: "RetrievalEvent",
      num_results: matchedDocs.length,
    };
  },
);

// ---------------------------------------------------------------------------
// Step 3: Generate a response
// ---------------------------------------------------------------------------
wf.addStep(
  "generate",
  ["RetrievalEvent"],
  async (event: Record<string, any>, ctx: Context) => {
    /**
     * Compose a final answer from the retrieved documents.
     *
     * In production this step would send the query + retrieved context to an
     * LLM.  Here we simulate the response by concatenating document snippets.
     */

    // Read shared data from context.
    const originalQuery: string = await ctx.get("original_query");
    const retrievedDocs: CorpusDocument[] = await ctx.get("retrieved_docs");
    const numDocs: number = await ctx.get("num_docs_retrieved");

    console.log(`[generate] Building response from ${numDocs} document(s)`);

    // Simulate LLM generation by summarising the retrieved content.
    const contextBlock: string = retrievedDocs
      .map((doc: CorpusDocument) => `[${doc.title}]: ${doc.content}`)
      .join("\n\n");

    const simulatedResponse: string =
      `Based on ${numDocs} retrieved document(s), here is the answer to ` +
      `your question "${originalQuery}":\n\n${contextBlock}`;

    console.log("[generate] Response generated successfully");

    return {
      type: "blazen::StopEvent",
      result: {
        response: simulatedResponse,
        sources: numDocs,
      },
    };
  },
);

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
const result: JsWorkflowResult = await wf.run({
  query: "How does Blazen share context between steps?",
});

console.log("\n" + "=".repeat(60));
console.log("FINAL RESULT");
console.log("=".repeat(60));
console.log("Sources consulted:", result.data.sources);
console.log(`\n${result.data.response}`);
