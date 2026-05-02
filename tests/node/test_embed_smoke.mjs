/**
 * embed local embedding smoke tests.
 *
 * Gated on the BLAZEN_TEST_EMBED environment variable.
 * Only runs when the native binding is compiled with the `embed` feature.
 *
 * Build first:
 *   cd crates/blazen-node && npm install && npm run build -- --features embed
 */

import test from "ava";

import { EmbeddingModel } from "../../crates/blazen-node/index.js";

const BLAZEN_TEST_EMBED = process.env.BLAZEN_TEST_EMBED;

const T = BLAZEN_TEST_EMBED ? test : test.skip;

T("local embedding · embeds two texts and returns correct count with non-empty vectors", async (t) => {
  const model = EmbeddingModel.embed();
  const response = await model.embed(["hello world", "goodbye world"]);

  t.is(response.embeddings.length, 2, "expected 2 embedding vectors");

  for (const vec of response.embeddings) {
    t.truthy(Array.isArray(vec), "each embedding should be an array");
    t.truthy(vec.length > 0, "embedding vector should not be empty");
    t.truthy(
      vec.every((v) => typeof v === "number" && Number.isFinite(v)),
      "all values should be finite numbers",
    );
  }
});

T("local embedding · constructs with explicit options and embeds one text", async (t) => {
  const model = EmbeddingModel.embed({
    showDownloadProgress: false,
    maxBatchSize: 32,
  });

  t.truthy(model.modelId, "expected a non-empty model ID");
  t.truthy(model.dimensions > 0, "expected positive dimensions");

  const response = await model.embed(["single input text"]);

  t.is(response.embeddings.length, 1, "expected 1 embedding vector");
  t.is(
    response.embeddings[0].length,
    model.dimensions,
    "vector length should match model dimensions",
  );
});
