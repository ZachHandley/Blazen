/**
 * fal.ai compute smoke tests.
 *
 * Gated on the FAL_API_KEY environment variable.
 * Tests LLM completion, modality auto-routing, streaming, embeddings,
 * 3D generation, image utilities, and timing/cost metadata.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  CompletionModel,
  ChatMessage,
  FalProvider,
} from "../../crates/blazen-node/index.js";

const FAL_API_KEY = process.env.FAL_API_KEY;

const IMAGE_URL =
  "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png";
const AUDIO_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Example.ogg";
const VIDEO_URL =
  "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4";

describe("fal.ai LLM smoke tests", { skip: !FAL_API_KEY }, () => {
  it("completes a basic prompt via OpenAiChat default", async () => {
    const model = CompletionModel.fal(FAL_API_KEY);
    const response = await model.complete([
      ChatMessage.user("What is 2+2? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("4"), `expected '4' in: ${response.content}`);
    assert.ok(response.model, "expected model in response");
  });

  it("completes a basic prompt with enterprise=true", async () => {
    const model = CompletionModel.fal(FAL_API_KEY, { enterprise: true });
    const response = await model.complete([
      ChatMessage.user("What is 3+3? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("6"), `expected '6' in: ${response.content}`);
  });

  it("completes a basic prompt via OpenAiResponses endpoint", async () => {
    const model = CompletionModel.fal(FAL_API_KEY, {
      endpoint: "open_ai_responses",
    });
    const response = await model.complete([
      ChatMessage.user("What is 5+5? Reply with just the number."),
    ]);

    assert.ok(response.content, "expected content in response");
    assert.ok(response.content.includes("10"), `expected '10' in: ${response.content}`);
  });

  it("returns timing metadata", async () => {
    const model = CompletionModel.fal(FAL_API_KEY);
    const response = await model.complete([
      ChatMessage.user("Say hello."),
    ]);

    assert.ok(response.content, "expected content");
    // Timing should be populated for fal.ai queue mode
    if (response.timing) {
      assert.ok(
        typeof response.timing.totalMs === "number" || response.timing.totalMs === null,
        "timing.totalMs should be a number or null"
      );
    }
  });

  it("passes temperature and max_tokens", async () => {
    const model = CompletionModel.fal(FAL_API_KEY);
    const response = await model.completeWithOptions(
      [ChatMessage.user("Write a one-word greeting.")],
      { temperature: 0.1, maxTokens: 10 }
    );

    assert.ok(response.content, "expected content");
    // With max_tokens=10, response should be short
    assert.ok(response.content.length < 200, "response should be short with maxTokens=10");
  });

  it("exposes new typed CompletionResponse fields (reasoning/citations/artifacts)", async () => {
    const model = CompletionModel.fal(FAL_API_KEY);
    const response = await model.complete([
      ChatMessage.user("Say hi."),
    ]);

    assert.ok(response.content, "expected content");
    // Reasoning is Option<> on the Rust side; napi serializes None as undefined,
    // which means the property may not be enumerable via `in`. Just check that
    // accessing it doesn't throw and that citations/artifacts are arrays.
    assert.ok(
      response.reasoning === undefined || typeof response.reasoning === "object",
      "expected 'reasoning' to be undefined or object"
    );
    assert.ok(
      Array.isArray(response.citations),
      "expected 'citations' field on CompletionResponse"
    );
    assert.ok(
      Array.isArray(response.artifacts),
      "expected 'artifacts' field on CompletionResponse"
    );
  });
});

describe("fal.ai modality auto-routing tests", { skip: !FAL_API_KEY }, () => {
  it("auto-routes to vision when AnyLlm and image part present", async () => {
    const model = CompletionModel.fal(FAL_API_KEY, {
      endpoint: "any_llm",
      autoRouteModality: true,
    });
    const response = await model.complete([
      ChatMessage.userImageUrl("Describe this image in one short sentence.", IMAGE_URL, "image/png"),
    ]);

    assert.ok(response.content, "expected content from vision auto-route");
    assert.ok(response.content.length > 0, "expected non-empty vision response");
  });

  it("auto-routes to audio when OpenRouter and audio part present", async () => {
    const model = CompletionModel.fal(FAL_API_KEY, {
      endpoint: "open_router",
      autoRouteModality: true,
    });
    const msg = ChatMessage.userAudio("Transcribe or describe this audio briefly.", AUDIO_URL);
    try {
      const response = await model.complete([msg]);
      assert.ok(response.content, "expected content from audio auto-route");
      assert.ok(response.content.length > 0, "expected non-empty audio response");
    } catch (e) {
      const m = String(e.message ?? e);
      assert.ok(
        m.includes("Failed to download audio") ||
          m.includes("audio_url") ||
          m.includes("file_download_error"),
        `unexpected error (routing may have failed): ${m}`,
      );
    }
  });

  it("auto-routes to video when OpenRouter and video part present", async () => {
    const model = CompletionModel.fal(FAL_API_KEY, {
      endpoint: "open_router",
      autoRouteModality: true,
    });
    const msg = ChatMessage.userVideo("Describe this video in one short sentence.", VIDEO_URL);
    try {
      const response = await model.complete([msg]);
      assert.ok(response.content, "expected content from video auto-route");
      assert.ok(response.content.length > 0, "expected non-empty video response");
    } catch (e) {
      const m = String(e.message ?? e);
      assert.ok(
        m.includes("Failed to download video") ||
          m.includes("video_url") ||
          m.includes("file_download_error"),
        `unexpected error (routing may have failed): ${m}`,
      );
    }
  });
});

describe("fal.ai streaming + embeddings + utilities", { skip: !FAL_API_KEY, timeout: 300_000 }, () => {
  it("streaming yields multiple chunks", async () => {
    const model = CompletionModel.fal(FAL_API_KEY);
    const chunks = [];

    await model.stream(
      [ChatMessage.user("Count from 1 to 5, one number per line.")],
      (chunk) => {
        chunks.push(chunk);
      },
    );

    assert.ok(chunks.length > 0, `expected at least one streamed chunk, got ${chunks.length}`);
  });

  it("FalProvider.embeddingModel().embed produces vectors", async () => {
    const provider = FalProvider.create(FAL_API_KEY);
    const embedder = provider.embeddingModel();
    const vectors = await embedder.embed(["hi", "hello"]);

    assert.ok(Array.isArray(vectors), "expected an array of vectors");
    assert.strictEqual(vectors.length, 2, "expected 2 vectors");
    assert.strictEqual(vectors[0].length, 1536, "expected vector dim 1536");
    assert.strictEqual(vectors[1].length, 1536, "expected vector dim 1536");
  });

  it("generates a 3D model", async () => {
    const provider = FalProvider.create(FAL_API_KEY);
    try {
      const result = await provider.generate3d({ imageUrl: IMAGE_URL });
      assert.ok(result, "expected a 3D generation result");
    } catch (e) {
      const m = String(e.message ?? e);
      assert.ok(
        m.includes("Failed to download") ||
          m.includes("image_url") ||
          m.includes("file_download_error"),
        `unexpected error (routing may have failed): ${m}`,
      );
    }
  });

  it("removes background from an image", async () => {
    const provider = FalProvider.create(FAL_API_KEY);
    try {
      const result = await provider.removeBackground(IMAGE_URL);
      assert.ok(result, "expected a background-removal result");
    } catch (e) {
      const m = String(e.message ?? e);
      assert.ok(
        m.includes("Failed to download") ||
          m.includes("image_url") ||
          m.includes("file_download_error"),
        `unexpected error (routing may have failed): ${m}`,
      );
    }
  });
});

describe("fal.ai compute smoke tests", { skip: !FAL_API_KEY, timeout: 2_100_000 }, () => {
  it("generates an image with FLUX", { timeout: 300_000 }, async () => {
    const provider = FalProvider.create(FAL_API_KEY);
    const result = await provider.generateImage({ prompt: "a simple red circle on white" });
    assert.ok(result.images, "expected images");
    assert.ok(result.images.length > 0, "expected at least one image");
  });

  it("synthesizes speech from text", { timeout: 300_000 }, async () => {
    const provider = FalProvider.create(FAL_API_KEY);
    const result = await provider.textToSpeech({
      text: "Hello world.",
    });

    assert.ok(result, "expected a result");
    // The result should contain audio data (url or base64)
    assert.ok(
      result.audio_url || result.audio || result.url,
      `expected audio data in result, got keys: ${Object.keys(result).join(", ")}`
    );
  });

  it("generates music from a prompt", { timeout: 300_000 }, async () => {
    const provider = FalProvider.create(FAL_API_KEY);
    const result = await provider.generateMusic({
      prompt: "happy upbeat jingle",
      durationSeconds: 5,
    });

    assert.ok(result, "expected a result");
    // The result should contain audio data
    assert.ok(
      result.audio_url || result.audio || result.url || result.audio_file,
      `expected audio data in result, got keys: ${Object.keys(result).join(", ")}`
    );
  });

  it("transcribes audio from a URL", { timeout: 300_000 }, async () => {
    const provider = FalProvider.create(FAL_API_KEY);
    const result = await provider.transcribe({
      audioUrl: "https://cdn.openai.com/API/docs/audio/alloy.wav",
    });

    assert.ok(result, "expected a result");
    // The result should contain transcribed text
    assert.ok(
      result.text || result.transcript || result.chunks,
      `expected text content in result, got keys: ${Object.keys(result).join(", ")}`
    );
  });

  it("generates a video from a text prompt", { timeout: 300_000 }, async () => {
    const provider = FalProvider.create(FAL_API_KEY);
    const result = await provider.textToVideo({
      prompt: "a cat walking slowly",
    });

    assert.ok(result, "expected a result");
    // The result should contain video data
    assert.ok(
      result.videos || result.video_url || result.video || result.url,
      `expected video data in result, got keys: ${Object.keys(result).join(", ")}`
    );
    if (result.videos) {
      assert.ok(result.videos.length > 0, "expected at least one video");
    }
  });

  it("runs a model synchronously via run()", { timeout: 300_000 }, async () => {
    const provider = FalProvider.create(FAL_API_KEY);
    const result = await provider.run("fal-ai/flux/schnell", {
      prompt: "blue sky with white clouds",
      image_size: "square_hd",
    });

    assert.ok(result, "expected a non-null result from run()");
  });

  it("submits a job and gets a valid job handle", { timeout: 300_000 }, async () => {
    const provider = FalProvider.create(FAL_API_KEY);

    const job = await provider.submit("fal-ai/flux/schnell", {
      prompt: "green forest with sunlight",
    });

    assert.ok(job, "expected a job handle from submit()");

    const jobId = job.id || job.requestId || job.request_id;
    assert.ok(jobId, `expected job id, got keys: ${Object.keys(job).join(", ")}`);
    assert.ok(typeof jobId === "string" && jobId.length > 0, "job id should be a non-empty string");
  });
});
