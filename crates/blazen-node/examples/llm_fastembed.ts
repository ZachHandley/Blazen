/**
 * Local embedding with Blazen's fastembed backend.
 *
 * Demonstrates using Blazen's EmbeddingModel with the fastembed (ONNX Runtime)
 * backend to produce embedding vectors entirely on-device -- no API key or
 * network access required.
 *
 * The default model is `BAAI/bge-small-en-v1.5` (384 dimensions). On the
 * first run fastembed will download and cache the ONNX weights automatically.
 *
 * Run with: npx tsx llm_fastembed.ts
 */

import { EmbeddingModel } from "blazen";

// ---------------------------------------------------------------------------
// Create a local fastembed model with default options.
// ---------------------------------------------------------------------------

const model: EmbeddingModel = EmbeddingModel.fastembed();

console.log(`Model ID:    ${model.modelId}`);
console.log(`Dimensions:  ${model.dimensions}`);

// ---------------------------------------------------------------------------
// Embed two sample strings.
// ---------------------------------------------------------------------------

const texts: string[] = [
  "Blazen makes LLM orchestration fast and type-safe.",
  "Rust-powered embeddings run entirely on-device.",
];

console.log(`\nEmbedding ${texts.length} texts...\n`);

const response = await model.embed(texts);

console.log(`Embeddings returned: ${response.embeddings.length}`);

for (let i = 0; i < response.embeddings.length; i++) {
  const vec = response.embeddings[i];
  const preview = vec.slice(0, 5).map((v: number) => v.toFixed(6)).join(", ");
  console.log(`  [${i}] length=${vec.length}  first 5: [${preview}, ...]`);
}

console.log("\nDone.");
