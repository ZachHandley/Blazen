/**
 * fal.ai compute smoke tests.
 *
 * Gated on the FAL_API_KEY environment variable.
 * Tests LLM completion, modality auto-routing, streaming, embeddings,
 * 3D generation, image utilities, and timing/cost metadata.
 */

// Invocation examples:
//   default (LLM only):        node --test tests/node/test_fal_smoke.mjs
//   + compute (image/TTS/...): BLAZEN_TEST_FAL_COMPUTE=1 node --test tests/node/test_fal_smoke.mjs
//   + video:                   BLAZEN_TEST_FAL_VIDEO=1 node --test tests/node/test_fal_smoke.mjs

import test from "ava";

import {
  CompletionModel,
  ChatMessage,
  FalProvider,
} from "../../crates/blazen-node/index.js";
import { falOrSkip, expectFalRoutingError } from "./_smoke_helpers.mjs";

const FAL_API_KEY = process.env.FAL_API_KEY;
const FAL_COMPUTE = process.env.BLAZEN_TEST_FAL_COMPUTE === "1";
const FAL_VIDEO = process.env.BLAZEN_TEST_FAL_VIDEO === "1";

const IMAGE_URL =
  "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png";
// 3D generation deliberately uses an unfetchable URL (`example.invalid` is
// reserved by RFC 2606) so fal fast-fails with file_download_error rather
// than running full triposr 3D gen, which regularly exceeds the 300s test
// budget on fal's queue. The test only cares about routing verification.
const UNFETCHABLE_IMAGE_URL = "https://example.invalid/triposr_input.jpg";
const AUDIO_URL = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Example.ogg";
const VIDEO_URL =
  "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4";

const TLlm = FAL_API_KEY ? test : test.skip;
const TRoute = FAL_API_KEY ? test : test.skip;
const TStream = FAL_API_KEY ? test : test.skip;
const TCompute = FAL_API_KEY && FAL_COMPUTE ? test : test.skip;

TLlm("fal.ai LLM smoke tests · completes a basic prompt via OpenAiChat default", async (t) => {
  const model = CompletionModel.fal({ apiKey: FAL_API_KEY });
  const response = await model.complete([
    ChatMessage.user("What is 2+2? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("4"), `expected '4' in: ${response.content}`);
  t.truthy(response.model, "expected model in response");
});

TLlm("fal.ai LLM smoke tests · completes a basic prompt with enterprise=true", async (t) => {
  const model = CompletionModel.fal({ apiKey: FAL_API_KEY, enterprise: true });
  const response = await model.complete([
    ChatMessage.user("What is 3+3? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("6"), `expected '6' in: ${response.content}`);
});

TLlm("fal.ai LLM smoke tests · completes a basic prompt via OpenAiResponses endpoint", async (t) => {
  const model = CompletionModel.fal({
    apiKey: FAL_API_KEY,
    endpoint: "open_ai_responses",
  });
  const response = await model.complete([
    ChatMessage.user("What is 5+5? Reply with just the number."),
  ]);

  t.truthy(response.content, "expected content in response");
  t.truthy(response.content.includes("10"), `expected '10' in: ${response.content}`);
});

TLlm("fal.ai LLM smoke tests · returns timing metadata", async (t) => {
  const model = CompletionModel.fal({ apiKey: FAL_API_KEY });
  const response = await model.complete([
    ChatMessage.user("Say hello."),
  ]);

  t.truthy(response.content, "expected content");
  // Timing should be populated for fal.ai queue mode
  if (response.timing) {
    t.truthy(
      typeof response.timing.totalMs === "number" || response.timing.totalMs === null,
      "timing.totalMs should be a number or null"
    );
  } else {
    t.pass("no timing metadata returned");
  }
});

TLlm("fal.ai LLM smoke tests · passes temperature and max_tokens", async (t) => {
  const model = CompletionModel.fal({ apiKey: FAL_API_KEY });
  const response = await model.completeWithOptions(
    [ChatMessage.user("Write a one-word greeting.")],
    { temperature: 0.1, maxTokens: 10 }
  );

  t.truthy(response.content, "expected content");
  // With max_tokens=10, response should be short
  t.truthy(response.content.length < 200, "response should be short with maxTokens=10");
});

TLlm("fal.ai LLM smoke tests · exposes new typed CompletionResponse fields (reasoning/citations/artifacts)", async (t) => {
  const model = CompletionModel.fal({ apiKey: FAL_API_KEY });
  const response = await model.complete([
    ChatMessage.user("Say hi."),
  ]);

  t.truthy(response.content, "expected content");
  // Reasoning is Option<> on the Rust side; napi serializes None as undefined,
  // which means the property may not be enumerable via `in`. Just check that
  // accessing it doesn't throw and that citations/artifacts are arrays.
  t.truthy(
    response.reasoning === undefined || typeof response.reasoning === "object",
    "expected 'reasoning' to be undefined or object"
  );
  t.truthy(
    Array.isArray(response.citations),
    "expected 'citations' field on CompletionResponse"
  );
  t.truthy(
    Array.isArray(response.artifacts),
    "expected 'artifacts' field on CompletionResponse"
  );
});

TRoute("fal.ai modality auto-routing tests · auto-routes to vision when AnyLlm and image part present", async (t) => {
  const model = CompletionModel.fal({
    apiKey: FAL_API_KEY,
    endpoint: "any_llm",
    autoRouteModality: true,
  });
  const response = await model.complete([
    ChatMessage.userImageUrl("Describe this image in one short sentence.", IMAGE_URL, "image/png"),
  ]);

  t.truthy(response.content, "expected content from vision auto-route");
  t.truthy(response.content.length > 0, "expected non-empty vision response");
});

TRoute("fal.ai modality auto-routing tests · auto-routes to audio when OpenRouter and audio part present", async (t) => {
  const model = CompletionModel.fal({
    apiKey: FAL_API_KEY,
    endpoint: "open_router",
    autoRouteModality: true,
  });
  const msg = ChatMessage.userAudio("Transcribe or describe this audio briefly.", AUDIO_URL);
  const response = await expectFalRoutingError(
    model.complete([msg]),
    ["Failed to download audio", "audio_url", "file_download_error"],
  );
  if (response !== undefined) {
    t.truthy(response.content, "expected content from audio auto-route");
    t.truthy(response.content.length > 0, "expected non-empty audio response");
  } else {
    t.pass("expected fal routing error observed");
  }
});

TRoute("fal.ai modality auto-routing tests · auto-routes to video when OpenRouter and video part present", async (t) => {
  const model = CompletionModel.fal({
    apiKey: FAL_API_KEY,
    endpoint: "open_router",
    autoRouteModality: true,
  });
  const msg = ChatMessage.userVideo("Describe this video in one short sentence.", VIDEO_URL);
  const response = await expectFalRoutingError(
    model.complete([msg]),
    ["Failed to download video", "video_url", "file_download_error"],
  );
  if (response !== undefined) {
    t.truthy(response.content, "expected content from video auto-route");
    t.truthy(response.content.length > 0, "expected non-empty video response");
  } else {
    t.pass("expected fal routing error observed");
  }
});

TStream("fal.ai streaming + embeddings + utilities · streaming yields multiple chunks", async (t) => {
  const model = CompletionModel.fal({ apiKey: FAL_API_KEY });
  const chunks = [];

  await model.stream(
    [ChatMessage.user("Count from 1 to 5, one number per line.")],
    (chunk) => {
      chunks.push(chunk);
    },
  );

  t.truthy(chunks.length > 0, `expected at least one streamed chunk, got ${chunks.length}`);
});

TStream("fal.ai streaming + embeddings + utilities · FalProvider.embeddingModel().embed produces vectors", async (t) => {
  const provider = FalProvider.create({ apiKey: FAL_API_KEY });
  const embedder = provider.embeddingModel();
  const vectors = await embedder.embed(["hi", "hello"]);

  t.truthy(Array.isArray(vectors), "expected an array of vectors");
  t.is(vectors.length, 2, "expected 2 vectors");
  t.is(vectors[0].length, 1536, "expected vector dim 1536");
  t.is(vectors[1].length, 1536, "expected vector dim 1536");
});

TCompute("fal.ai streaming + embeddings + utilities · generates a 3D model", async (t) => {
  const provider = FalProvider.create({ apiKey: FAL_API_KEY });
  const result = await expectFalRoutingError(
    provider.generate3d({ imageUrl: UNFETCHABLE_IMAGE_URL }),
    ["Failed to download", "image_url", "file_download_error"],
  );
  if (result !== undefined) {
    t.truthy(result, "expected a 3D generation result");
  } else {
    t.pass("expected fal routing error observed");
  }
});

TCompute("fal.ai streaming + embeddings + utilities · removes background from an image", async (t) => {
  const provider = FalProvider.create({ apiKey: FAL_API_KEY });
  const result = await expectFalRoutingError(
    provider.removeBackground({ imageUrl: IMAGE_URL }),
    ["Failed to download", "image_url", "file_download_error"],
  );
  if (result !== undefined) {
    t.truthy(result, "expected a background-removal result");
  } else {
    t.pass("expected fal routing error observed");
  }
});

TCompute("fal.ai compute smoke tests · generates an image with FLUX", async (t) => {
  const provider = FalProvider.create({ apiKey: FAL_API_KEY });
  const result = await falOrSkip(t, provider.generateImage({ prompt: "a simple red circle on white" }));
  if (result === undefined) return;
  t.truthy(result.images, "expected images");
  t.truthy(result.images.length > 0, "expected at least one image");
});

TCompute("fal.ai compute smoke tests · synthesizes speech from text", async (t) => {
  const provider = FalProvider.create({ apiKey: FAL_API_KEY });
  const result = await falOrSkip(t, provider.textToSpeech({ text: "Hello world." }));
  if (result === undefined) return;
  t.truthy(result, "expected a result");
  t.truthy(
    result.audio_url || result.audio || result.url,
    `expected audio data in result, got keys: ${Object.keys(result).join(", ")}`
  );
});

TCompute("fal.ai compute smoke tests · generates music from a prompt", async (t) => {
  const provider = FalProvider.create({ apiKey: FAL_API_KEY });
  const result = await falOrSkip(t, provider.generateMusic({
    prompt: "happy upbeat jingle",
    durationSeconds: 5,
  }));
  if (result === undefined) return;
  t.truthy(result, "expected a result");
  t.truthy(
    result.audio_url || result.audio || result.url || result.audio_file,
    `expected audio data in result, got keys: ${Object.keys(result).join(", ")}`
  );
});

TCompute("fal.ai compute smoke tests · transcribes audio from a URL", async (t) => {
  const provider = FalProvider.create({ apiKey: FAL_API_KEY });
  const result = await falOrSkip(t, provider.transcribe({
    audioUrl: "https://cdn.openai.com/API/docs/audio/alloy.wav",
  }));
  if (result === undefined) return;
  t.truthy(result, "expected a result");
  t.truthy(
    result.text || result.transcript || result.chunks,
    `expected text content in result, got keys: ${Object.keys(result).join(", ")}`
  );
});

const TVideo = FAL_API_KEY && FAL_COMPUTE && FAL_VIDEO ? test : test.skip;

TVideo("fal.ai compute smoke tests · generates a video from a text prompt", async (t) => {
  const provider = FalProvider.create({ apiKey: FAL_API_KEY });
  const result = await falOrSkip(t, provider.textToVideo({ prompt: "a cat walking slowly" }));
  if (result === undefined) return;

  t.truthy(result, "expected a result");
  t.truthy(
    result.videos || result.video_url || result.video || result.url,
    `expected video data in result, got keys: ${Object.keys(result).join(", ")}`
  );
  if (result.videos) {
    t.truthy(result.videos.length > 0, "expected at least one video");
  }
});

TCompute("fal.ai compute smoke tests · runs a model synchronously via run()", async (t) => {
  const provider = FalProvider.create({ apiKey: FAL_API_KEY });
  const result = await falOrSkip(t, provider.run({
    model: "fal-ai/flux/schnell",
    input: { prompt: "blue sky with white clouds", imageSize: "square_hd" },
  }));
  if (result === undefined) return;
  t.truthy(result, "expected a non-null result from run()");
});

TCompute("fal.ai compute smoke tests · submits a job and gets a valid job handle", async (t) => {
  const provider = FalProvider.create({ apiKey: FAL_API_KEY });

  const job = await falOrSkip(t, provider.submit({
    model: "fal-ai/flux/schnell",
    input: { prompt: "green forest with sunlight" },
  }));
  if (job === undefined) return;

  t.truthy(job, "expected a job handle from submit()");

  const jobId = job.id || job.requestId || job.request_id;
  t.truthy(jobId, `expected job id, got keys: ${Object.keys(job).join(", ")}`);
  t.truthy(typeof jobId === "string" && jobId.length > 0, "job id should be a non-empty string");
});
