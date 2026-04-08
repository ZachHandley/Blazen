//! LLM RAG (Retrieval-Augmented Generation) workflow example.
//!
//! Demonstrates:
//! - A multi-step RAG pipeline structure
//! - Using shared `Context` state to pass data between steps
//! - How LLM integration would work (uses mock responses)
//!
//! Run with: `cargo run -p blazen --example llm_rag_workflow`
//!
//! Note: This example uses simulated LLM responses. To use real providers,
//! enable the `openai` or `anthropic` feature and supply API keys.

// Steps must be async for the #[step] macro even when they don't await.
#![allow(clippy::unused_async, clippy::cast_possible_truncation)]

use blazen::prelude::*;

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

/// Event carrying the user's query and retrieval parameters.
#[derive(Debug, Clone, Serialize, Deserialize, Event)]
struct QueryEvent {
    query: String,
    top_k: usize,
}

/// Event carrying retrieved documents for the query.
#[derive(Debug, Clone, Serialize, Deserialize, Event)]
struct RetrievalEvent {
    query: String,
    documents: Vec<String>,
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

/// Parse the raw input into a structured query.
#[step]
async fn parse_query(event: StartEvent, _ctx: Context) -> Result<QueryEvent, WorkflowError> {
    let query = event.data["query"]
        .as_str()
        .unwrap_or("What is Rust?")
        .to_string();
    let top_k = event.data["top_k"].as_u64().unwrap_or(3) as usize;

    println!("[parse_query] query={query:?}  top_k={top_k}");
    Ok(QueryEvent { query, top_k })
}

/// Simulate document retrieval (in a real app, this would query a vector DB).
#[step]
async fn retrieve(event: QueryEvent, ctx: Context) -> Result<RetrievalEvent, WorkflowError> {
    println!("[retrieve] Searching for top-{} documents...", event.top_k);

    // Simulated document corpus.
    let corpus: &[&str] = &[
        "Rust is a systems programming language focused on safety and performance.",
        "The Rust borrow checker ensures memory safety without garbage collection.",
        "Tokio is an async runtime for Rust that provides work-stealing scheduling.",
        "Cargo is Rust's package manager and build tool.",
        "Rust's type system prevents data races at compile time.",
    ];

    // Simulate retrieval: pick top_k documents that contain query keywords.
    let query_lower = event.query.to_lowercase();
    let mut documents: Vec<String> = corpus
        .iter()
        .filter(|doc| {
            let doc_lower = doc.to_lowercase();
            query_lower
                .split_whitespace()
                .any(|word| doc_lower.contains(word))
        })
        .take(event.top_k)
        .map(|s| (*s).to_string())
        .collect();

    // Fall back to first document if nothing matched.
    if documents.is_empty() {
        documents.push(corpus[0].to_string());
    }

    // Store documents in context for downstream steps.
    ctx.set("retrieved_count", documents.len()).await;

    println!("[retrieve] Found {} documents", documents.len());
    Ok(RetrievalEvent {
        query: event.query,
        documents,
    })
}

/// Simulate LLM generation (in a real app, this would call an LLM provider).
#[step]
async fn generate(event: RetrievalEvent, ctx: Context) -> Result<StopEvent, WorkflowError> {
    println!(
        "[generate] Generating answer from {} documents...",
        event.documents.len()
    );

    // Build a mock "prompt" from the retrieved documents.
    let _context_str = event
        .documents
        .iter()
        .enumerate()
        .map(|(i, doc)| format!("[{}] {}", i + 1, doc))
        .collect::<Vec<_>>()
        .join("\n");

    // In a real application you would do something like:
    //
    //   let model = OpenAiProvider::new(&api_key);
    //   let response = model.complete(CompletionRequest::new(vec![
    //       ChatMessage::system("Answer based on the following context:"),
    //       ChatMessage::user(&format!("{context_str}\n\nQuestion: {}", event.query)),
    //   ])).await?;

    // Simulated LLM response.
    let answer = format!(
        "Based on the retrieved documents, here is the answer to \"{}\": {}",
        event.query,
        event
            .documents
            .first()
            .unwrap_or(&"No information available.".to_string())
    );

    let retrieved_count: usize = ctx.get("retrieved_count").await.unwrap_or(0);
    println!("[generate] Used {retrieved_count} sources");

    Ok(StopEvent {
        result: serde_json::json!({
            "query": event.query,
            "answer": answer,
            "sources": event.documents,
            "source_count": retrieved_count,
        }),
    })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let workflow = WorkflowBuilder::new("rag_pipeline")
        .step(parse_query_registration())
        .step(retrieve_registration())
        .step(generate_registration())
        .build()?;

    println!("=== RAG Pipeline ===\n");

    let result = workflow
        .run(serde_json::json!({
            "query": "What is Rust's borrow checker?",
            "top_k": 3
        }))
        .await?
        .result()
        .await?
        .event;

    if let Some(stop) = result.downcast_ref::<StopEvent>() {
        println!("\n=== Result ===");
        println!("{}", serde_json::to_string_pretty(&stop.result)?);
    }

    Ok(())
}
