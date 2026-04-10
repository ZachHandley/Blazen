/**
 * Local LLM inference with Blazen's mistral.rs backend.
 *
 * Demonstrates using Blazen's CompletionModel with the mistral.rs engine to
 * run LLM inference entirely on-device -- no API key or network access
 * required (after the initial model download from HuggingFace Hub).
 *
 * The default model used here is a GGUF-quantised Mistral-7B-Instruct variant.
 * Replace with any HuggingFace model ID or local GGUF path.
 *
 * Run with: npx tsx llm_mistralrs.ts
 */

import { CompletionModel, ChatMessage } from "blazen";

// ---------------------------------------------------------------------------
// 1. Create a local mistral.rs model.
// ---------------------------------------------------------------------------

const model: CompletionModel = CompletionModel.mistralrs({
  modelId: "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
});

console.log(`Model ID: ${model.modelId}`);
console.log();

// ---------------------------------------------------------------------------
// 2. Simple completion.
// ---------------------------------------------------------------------------

console.log("--- Simple Completion ---");

const response = await model.complete([
  ChatMessage.system("You are a helpful assistant. Answer concisely in 1-2 sentences."),
  ChatMessage.user("What is 2+2?"),
]);

console.log(`Response: ${response.content}`);
console.log(`Model:    ${response.model}`);
if (response.usage) {
  console.log(`Tokens:   ${JSON.stringify(response.usage)}`);
}
console.log();

// ---------------------------------------------------------------------------
// 3. Completion with options (temperature, maxTokens).
// ---------------------------------------------------------------------------

console.log("--- Completion With Options ---");

const storyResponse = await model.completeWithOptions(
  [
    ChatMessage.system("You are a creative storyteller."),
    ChatMessage.user("Tell me a very short story about a robot learning to cook."),
  ],
  { temperature: 0.9, maxTokens: 256 },
);

console.log(`Response: ${storyResponse.content}`);
console.log();

// ---------------------------------------------------------------------------
// 4. Streaming completion.
// ---------------------------------------------------------------------------

console.log("--- Streaming ---");

let chunksReceived = 0;
await model.stream(
  [ChatMessage.user("Explain why the sky is blue in 2-3 sentences.")],
  (chunk) => {
    if (chunk.delta) {
      process.stdout.write(chunk.delta);
    }
    chunksReceived++;
  },
);

console.log();
console.log(`\n(received ${chunksReceived} chunks)`);
console.log();
console.log("Done.");
