/**
 * fastembed local embedding smoke tests.
 *
 * Gated on the BLAZEN_TEST_FASTEMBED environment variable.
 * Only runs when the native binding is compiled with the `fastembed` feature.
 *
 * Build first:
 *   cd crates/blazen-node && npm install && npm run build -- --features fastembed
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { EmbeddingModel } from "../../crates/blazen-node/index.js";

const BLAZEN_TEST_FASTEMBED = process.env.BLAZEN_TEST_FASTEMBED;

describe("fastembed local embedding", { skip: !BLAZEN_TEST_FASTEMBED }, () => {
  it("embeds two texts and returns correct count with non-empty vectors", async () => {
    const model = EmbeddingModel.fastembed();
    const response = await model.embed(["hello world", "goodbye world"]);

    assert.strictEqual(response.embeddings.length, 2, "expected 2 embedding vectors");

    for (const vec of response.embeddings) {
      assert.ok(Array.isArray(vec), "each embedding should be an array");
      assert.ok(vec.length > 0, "embedding vector should not be empty");
      assert.ok(
        vec.every((v) => typeof v === "number" && Number.isFinite(v)),
        "all values should be finite numbers",
      );
    }
  });

  it("constructs with explicit options and embeds one text", async () => {
    const model = EmbeddingModel.fastembed({
      showDownloadProgress: false,
      maxBatchSize: 32,
    });

    assert.ok(model.modelId, "expected a non-empty model ID");
    assert.ok(model.dimensions > 0, "expected positive dimensions");

    const response = await model.embed(["single input text"]);

    assert.strictEqual(response.embeddings.length, 1, "expected 1 embedding vector");
    assert.strictEqual(
      response.embeddings[0].length,
      model.dimensions,
      "vector length should match model dimensions",
    );
  });
});
