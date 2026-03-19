/**
 * Comprehensive fal.ai compute smoke tests.
 *
 * Gated on the FAL_API_KEY environment variable.
 * Tests image generation, text-to-speech, music generation, transcription,
 * text-to-video, raw compute, and job lifecycle.
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import { FalProvider } from "../../crates/blazen-node/index.js";

const FAL_API_KEY = process.env.FAL_API_KEY;

// -- Image Generation --------------------------------------------------------

describe("fal.ai image generation smoke tests", { skip: !FAL_API_KEY }, () => {
  it("generates an image from a text prompt", async () => {
    const provider = FalProvider.create(FAL_API_KEY);
    const result = await provider.generateImage({
      prompt: "a red circle on a white background",
    });

    assert.ok(result, "expected a result");
    assert.ok(result.images, "expected images in result");
    assert.ok(result.images.length > 0, "expected at least one image");
  });
});

// -- Text to Speech ----------------------------------------------------------

describe("fal.ai text-to-speech smoke tests", { skip: !FAL_API_KEY }, () => {
  it("synthesizes speech from text", async () => {
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
});

// -- Music Generation --------------------------------------------------------

describe("fal.ai music generation smoke tests", { skip: !FAL_API_KEY }, () => {
  it("generates music from a prompt", async () => {
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
});

// -- Transcription -----------------------------------------------------------

describe("fal.ai transcription smoke tests", { skip: !FAL_API_KEY }, () => {
  it("transcribes audio from a URL", async () => {
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
});

// -- Text to Video -----------------------------------------------------------

describe("fal.ai text-to-video smoke tests", { skip: !FAL_API_KEY }, () => {
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
});

// -- Raw Compute (run) -------------------------------------------------------

describe("fal.ai raw compute smoke tests", { skip: !FAL_API_KEY }, () => {
  it("runs a model synchronously via run()", async () => {
    const provider = FalProvider.create(FAL_API_KEY);
    const result = await provider.run("fal-ai/flux/schnell", {
      prompt: "blue sky with white clouds",
      image_size: "square_hd",
    });

    assert.ok(result, "expected a non-null result from run()");
  });
});

// -- Job Lifecycle (submit + status) -----------------------------------------

describe("fal.ai job lifecycle smoke tests", { skip: !FAL_API_KEY }, () => {
  it("submits a job and gets a valid job handle", async () => {
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
