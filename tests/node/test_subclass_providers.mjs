/**
 * Subclassable provider integration tests.
 *
 * Exercises the subclassing surface for CompletionModel, EmbeddingModel,
 * capability providers (TTS, Music, Image, Video, 3D, BackgroundRemoval,
 * Voice), MemoryBackend, ModelManager, CustomProvider, and pricing
 * registration/lookup.
 *
 * No API keys required -- all tests run locally without network calls.
 *
 * Build the native binding first:
 *   pnpm --filter blazen run build
 */

import { describe, it } from "node:test";
import assert from "node:assert/strict";

import {
  ChatMessage,
  CompletionModel,
  CustomProvider,
  EmbeddingModel,
  InMemoryBackend,
  Memory,
  ModelManager,
  MemoryBackend,
  TTSProvider,
  MusicProvider,
  ImageProvider,
  VideoProvider,
  ThreeDProvider,
  BackgroundRemovalProvider,
  VoiceProvider,
  registerPricing,
  lookupPricing,
} from "../../crates/blazen-node/index.js";

// ===========================================================================
// CompletionModel subclassing
// ===========================================================================

describe("CompletionModel subclassing", () => {
  it("can be subclassed with modelId via super(config)", () => {
    class MockLLM extends CompletionModel {
      constructor() {
        super({ modelId: "mock-llm", contextLength: 4096 });
      }
    }
    const model = new MockLLM();
    assert.equal(model.modelId, "mock-llm");
  });

  it("can be subclassed with no config", () => {
    class BareLLM extends CompletionModel {
      constructor() {
        super();
      }
    }
    const model = new BareLLM();
    // With no config, modelId defaults to empty string
    assert.equal(model.modelId, "");
  });

  it("can be subclassed with all config fields", () => {
    class FullConfigLLM extends CompletionModel {
      constructor() {
        super({
          modelId: "full-model",
          contextLength: 8192,
          baseUrl: "https://example.com/v1",
          vramEstimateBytes: 4_000_000_000,
          maxOutputTokens: 2048,
        });
      }
    }
    const model = new FullConfigLLM();
    assert.equal(model.modelId, "full-model");
  });

  it("complete() throws 'subclass must override' for subclassed instances", async () => {
    class NoOverrideLLM extends CompletionModel {
      constructor() {
        super({ modelId: "no-override" });
      }
    }
    const model = new NoOverrideLLM();
    await assert.rejects(
      () => model.complete([ChatMessage.user("test")]),
      /subclass must override/
    );
  });

  it("completeWithOptions() throws for subclassed instances without override", async () => {
    class NoOverrideLLM extends CompletionModel {
      constructor() {
        super({ modelId: "no-override" });
      }
    }
    const model = new NoOverrideLLM();
    await assert.rejects(
      () =>
        model.completeWithOptions([ChatMessage.user("test")], {
          maxTokens: 10,
        }),
      /subclass must override/
    );
  });

  it("withRetry() throws for subclassed instances", () => {
    class SubLLM extends CompletionModel {
      constructor() {
        super({ modelId: "sub" });
      }
    }
    const model = new SubLLM();
    assert.throws(
      () => model.withRetry({ maxRetries: 3 }),
      /not supported on subclassed/
    );
  });

  it("withCache() throws for subclassed instances", () => {
    class SubLLM extends CompletionModel {
      constructor() {
        super({ modelId: "sub" });
      }
    }
    const model = new SubLLM();
    assert.throws(
      () => model.withCache({ ttlSeconds: 60 }),
      /not supported on subclassed/
    );
  });

  it("isLoaded() returns false for subclassed instances", async () => {
    class SubLLM extends CompletionModel {
      constructor() {
        super({ modelId: "sub" });
      }
    }
    const model = new SubLLM();
    const loaded = await model.isLoaded();
    assert.strictEqual(loaded, false);
  });

  it("vramBytes() returns null for subclassed instances", async () => {
    class SubLLM extends CompletionModel {
      constructor() {
        super({ modelId: "sub" });
      }
    }
    const model = new SubLLM();
    const bytes = await model.vramBytes();
    assert.strictEqual(bytes, null);
  });
});

// ===========================================================================
// EmbeddingModel subclassing
// ===========================================================================

describe("EmbeddingModel subclassing", () => {
  it("can be subclassed with modelId and dimensions", () => {
    class MockEmbed extends EmbeddingModel {
      constructor() {
        super({ modelId: "mock-embed", dimensions: 768 });
      }
    }
    const model = new MockEmbed();
    assert.equal(model.modelId, "mock-embed");
    assert.equal(model.dimensions, 768);
  });

  it("can be subclassed with no config", () => {
    class BareEmbed extends EmbeddingModel {
      constructor() {
        super();
      }
    }
    const model = new BareEmbed();
    assert.equal(model.modelId, "");
    assert.equal(model.dimensions, 0);
  });

  it("can be subclassed with baseUrl", () => {
    class RemoteEmbed extends EmbeddingModel {
      constructor() {
        super({
          modelId: "remote-embed",
          dimensions: 1536,
          baseUrl: "https://embed.example.com/v1",
        });
      }
    }
    const model = new RemoteEmbed();
    assert.equal(model.modelId, "remote-embed");
    assert.equal(model.dimensions, 1536);
  });

  it("embed() throws 'subclass must override' for subclassed instances", async () => {
    class NoOverrideEmbed extends EmbeddingModel {
      constructor() {
        super({ modelId: "no-override-embed", dimensions: 384 });
      }
    }
    const model = new NoOverrideEmbed();
    await assert.rejects(
      () => model.embed(["hello", "world"]),
      /subclass must override/
    );
  });
});

// ===========================================================================
// Capability providers -- TTSProvider
// ===========================================================================

describe("TTSProvider subclassing", () => {
  it("can be constructed with providerId", () => {
    const provider = new TTSProvider({ providerId: "mock-tts" });
    assert.equal(provider.providerId, "mock-tts");
  });

  it("can be constructed with baseUrl and vramEstimateBytes", () => {
    const provider = new TTSProvider({
      providerId: "local-tts",
      baseUrl: "http://localhost:5000",
      vramEstimateBytes: 2_000_000_000,
    });
    assert.equal(provider.providerId, "local-tts");
    assert.equal(provider.baseUrl, "http://localhost:5000");
    assert.equal(provider.vramEstimateBytes, 2_000_000_000);
  });

  it("textToSpeech() throws 'subclass must override' by default", async () => {
    const provider = new TTSProvider({ providerId: "base-tts" });
    await assert.rejects(
      () => provider.textToSpeech({ text: "hello" }),
      /subclass must override textToSpeech/
    );
  });
});

// ===========================================================================
// Capability providers -- MusicProvider
// ===========================================================================

describe("MusicProvider subclassing", () => {
  it("can be constructed with providerId", () => {
    const provider = new MusicProvider({ providerId: "mock-music" });
    assert.equal(provider.providerId, "mock-music");
  });

  it("generateMusic() throws 'subclass must override' by default", async () => {
    const provider = new MusicProvider({ providerId: "base-music" });
    await assert.rejects(
      () => provider.generateMusic({ prompt: "epic battle" }),
      /subclass must override generateMusic/
    );
  });

  it("generateSfx() throws 'subclass must override' by default", async () => {
    const provider = new MusicProvider({ providerId: "base-music" });
    await assert.rejects(
      () => provider.generateSfx({ prompt: "explosion" }),
      /subclass must override generateSfx/
    );
  });
});

// ===========================================================================
// Capability providers -- ImageProvider
// ===========================================================================

describe("ImageProvider subclassing", () => {
  it("can be constructed with providerId", () => {
    const provider = new ImageProvider({ providerId: "mock-image" });
    assert.equal(provider.providerId, "mock-image");
  });

  it("generateImage() throws 'subclass must override' by default", async () => {
    const provider = new ImageProvider({ providerId: "base-image" });
    await assert.rejects(
      () => provider.generateImage({ prompt: "cat" }),
      /subclass must override generateImage/
    );
  });

  it("upscaleImage() throws 'subclass must override' by default", async () => {
    const provider = new ImageProvider({ providerId: "base-image" });
    await assert.rejects(
      () => provider.upscaleImage({ image: "base64data" }),
      /subclass must override upscaleImage/
    );
  });
});

// ===========================================================================
// Capability providers -- VideoProvider
// ===========================================================================

describe("VideoProvider subclassing", () => {
  it("can be constructed with providerId", () => {
    const provider = new VideoProvider({ providerId: "mock-video" });
    assert.equal(provider.providerId, "mock-video");
  });

  it("textToVideo() throws 'subclass must override' by default", async () => {
    const provider = new VideoProvider({ providerId: "base-video" });
    await assert.rejects(
      () => provider.textToVideo({ prompt: "sunset" }),
      /subclass must override textToVideo/
    );
  });

  it("imageToVideo() throws 'subclass must override' by default", async () => {
    const provider = new VideoProvider({ providerId: "base-video" });
    await assert.rejects(
      () => provider.imageToVideo({ image: "base64data" }),
      /subclass must override imageToVideo/
    );
  });
});

// ===========================================================================
// Capability providers -- ThreeDProvider
// ===========================================================================

describe("ThreeDProvider subclassing", () => {
  it("can be constructed with providerId", () => {
    const provider = new ThreeDProvider({ providerId: "mock-3d" });
    assert.equal(provider.providerId, "mock-3d");
  });

  it("generate3d() throws 'subclass must override' by default", async () => {
    const provider = new ThreeDProvider({ providerId: "base-3d" });
    await assert.rejects(
      () => provider.generate3d({ prompt: "chair" }),
      /subclass must override generate3d/
    );
  });
});

// ===========================================================================
// Capability providers -- BackgroundRemovalProvider
// ===========================================================================

describe("BackgroundRemovalProvider subclassing", () => {
  it("can be constructed with providerId", () => {
    const provider = new BackgroundRemovalProvider({
      providerId: "mock-bgremoval",
    });
    assert.equal(provider.providerId, "mock-bgremoval");
  });

  it("removeBackground() throws 'subclass must override' by default", async () => {
    const provider = new BackgroundRemovalProvider({
      providerId: "base-bgremoval",
    });
    await assert.rejects(
      () => provider.removeBackground({ image: "base64data" }),
      /subclass must override removeBackground/
    );
  });
});

// ===========================================================================
// Capability providers -- VoiceProvider
// ===========================================================================

describe("VoiceProvider subclassing", () => {
  it("can be constructed with providerId", () => {
    const provider = new VoiceProvider({ providerId: "mock-voice" });
    assert.equal(provider.providerId, "mock-voice");
  });

  it("cloneVoice() throws 'subclass must override' by default", async () => {
    const provider = new VoiceProvider({ providerId: "base-voice" });
    await assert.rejects(
      () => provider.cloneVoice({ name: "Alice", samples: [] }),
      /subclass must override cloneVoice/
    );
  });

  it("listVoices() throws 'subclass must override' by default", async () => {
    const provider = new VoiceProvider({ providerId: "base-voice" });
    await assert.rejects(
      () => provider.listVoices(),
      /subclass must override listVoices/
    );
  });

  it("deleteVoice() throws 'subclass must override' by default", async () => {
    const provider = new VoiceProvider({ providerId: "base-voice" });
    await assert.rejects(
      () => provider.deleteVoice({ voiceId: "v1" }),
      /subclass must override deleteVoice/
    );
  });
});

// ===========================================================================
// Capability providers -- common getter coverage
// ===========================================================================

describe("capability provider common getters", () => {
  it("baseUrl returns null when not set", () => {
    const provider = new TTSProvider({ providerId: "no-url" });
    assert.strictEqual(provider.baseUrl, null);
  });

  it("vramEstimateBytes returns null when not set", () => {
    const provider = new ImageProvider({ providerId: "no-vram" });
    assert.strictEqual(provider.vramEstimateBytes, null);
  });
});

// ===========================================================================
// MemoryBackend subclassing
// ===========================================================================

describe("MemoryBackend subclassing", () => {
  it("can be constructed", () => {
    const backend = new MemoryBackend();
    assert.ok(backend);
  });

  it("can be subclassed", () => {
    class DictBackend extends MemoryBackend {
      constructor() {
        super();
        this.store = new Map();
      }
    }
    const backend = new DictBackend();
    assert.ok(backend);
    assert.ok(backend.store instanceof Map);
  });

  it("put() throws 'subclass must override' by default", async () => {
    const backend = new MemoryBackend();
    await assert.rejects(
      () => backend.put({ id: "x", text: "hi" }),
      /subclass must override put/
    );
  });

  it("get() throws 'subclass must override' by default", async () => {
    const backend = new MemoryBackend();
    await assert.rejects(
      () => backend.get("x"),
      /subclass must override get/
    );
  });

  it("delete() throws 'subclass must override' by default", async () => {
    const backend = new MemoryBackend();
    await assert.rejects(
      () => backend.delete("x"),
      /subclass must override delete/
    );
  });

  it("list() throws 'subclass must override' by default", async () => {
    const backend = new MemoryBackend();
    await assert.rejects(
      () => backend.list(),
      /subclass must override list/
    );
  });

  it("len() throws 'subclass must override' by default", async () => {
    const backend = new MemoryBackend();
    await assert.rejects(
      () => backend.len(),
      /subclass must override len/
    );
  });

  it("searchByBands() throws 'subclass must override' by default", async () => {
    const backend = new MemoryBackend();
    await assert.rejects(
      () => backend.searchByBands(["band1"], 5),
      /subclass must override searchByBands/
    );
  });
});

// ===========================================================================
// ModelManager
// ===========================================================================

describe("ModelManager", () => {
  it("can be created with budgetGb", () => {
    const manager = new ModelManager({ budgetGb: 24 });
    assert.ok(manager);
  });

  it("can be created with budgetBytes", () => {
    const manager = new ModelManager({ budgetBytes: 8_000_000_000 });
    assert.ok(manager);
  });

  it("throws when neither budgetGb nor budgetBytes is given", () => {
    assert.throws(
      () => new ModelManager({}),
      /must provide either budgetGb or budgetBytes/
    );
  });

  it("reports zero usedBytes when empty", async () => {
    const manager = new ModelManager({ budgetGb: 8 });
    const used = await manager.usedBytes();
    assert.strictEqual(used, 0);
  });

  it("reports full budget as availableBytes when empty", async () => {
    const manager = new ModelManager({ budgetGb: 8 });
    const available = await manager.availableBytes();
    assert.ok(available > 0);
  });

  it("status() returns empty array when no models registered", async () => {
    const manager = new ModelManager({ budgetGb: 8 });
    const statuses = await manager.status();
    assert.deepStrictEqual(statuses, []);
  });

  it("register() throws for remote (non-local) models", async () => {
    const manager = new ModelManager({ budgetGb: 8 });
    const model = CompletionModel.openai({ apiKey: "fake-key" });
    await assert.rejects(
      () => manager.register("gpt-4", model),
      /does not support local loading/
    );
  });
});

// ===========================================================================
// Pricing registration and lookup
// ===========================================================================

describe("pricing registration and lookup", () => {
  it("registerPricing and lookupPricing round-trip", () => {
    registerPricing("test-subclass-model", {
      inputPerMillion: 1.5,
      outputPerMillion: 3.0,
    });
    const pricing = lookupPricing("test-subclass-model");
    assert.ok(pricing, "expected pricing to be found");
    assert.strictEqual(pricing.inputPerMillion, 1.5);
    assert.strictEqual(pricing.outputPerMillion, 3.0);
  });

  it("lookupPricing returns null for unknown models", () => {
    const pricing = lookupPricing("definitely-not-a-real-model-xyz-12345");
    assert.strictEqual(pricing, null);
  });

  it("registerPricing with per_image and per_second", () => {
    registerPricing("multimodal-test-model", {
      inputPerMillion: 5.0,
      outputPerMillion: 15.0,
      perImage: 0.02,
      perSecond: 0.001,
    });
    const pricing = lookupPricing("multimodal-test-model");
    assert.ok(pricing, "expected pricing to be found");
    assert.strictEqual(pricing.inputPerMillion, 5.0);
    assert.strictEqual(pricing.outputPerMillion, 15.0);
  });

  it("registerPricing overwrites previous registration", () => {
    registerPricing("overwrite-test-model", {
      inputPerMillion: 1.0,
      outputPerMillion: 2.0,
    });
    registerPricing("overwrite-test-model", {
      inputPerMillion: 10.0,
      outputPerMillion: 20.0,
    });
    const pricing = lookupPricing("overwrite-test-model");
    assert.ok(pricing);
    assert.strictEqual(pricing.inputPerMillion, 10.0);
    assert.strictEqual(pricing.outputPerMillion, 20.0);
  });
});

// ===========================================================================
// CustomProvider (host-dispatch wrapper)
// ===========================================================================

describe("CustomProvider", () => {
  it("can wrap a host object with textToSpeech", () => {
    const host = {
      async textToSpeech(request) {
        return { audio: [], timing: {}, metadata: {} };
      },
    };
    const provider = new CustomProvider(host, { providerId: "test-tts" });
    assert.equal(provider.providerId, "test-tts");
  });

  it("defaults providerId to 'custom' when not specified", () => {
    const host = {};
    const provider = new CustomProvider(host);
    assert.equal(provider.providerId, "custom");
  });

  it("can wrap a host object with multiple capabilities", () => {
    const host = {
      async textToSpeech(request) {
        return {};
      },
      async generateImage(request) {
        return {};
      },
      async textToVideo(request) {
        return {};
      },
    };
    const provider = new CustomProvider(host, {
      providerId: "multi-cap",
    });
    assert.equal(provider.providerId, "multi-cap");
  });

  it("can wrap an empty host object (no capabilities)", () => {
    const host = {};
    const provider = new CustomProvider(host, { providerId: "empty" });
    assert.equal(provider.providerId, "empty");
  });
});

// ===========================================================================
// CompletionModel factory sanity (no network calls)
// ===========================================================================

describe("CompletionModel factories (construction only)", () => {
  it("openai factory sets modelId", () => {
    const model = CompletionModel.openai({ apiKey: "fake-key" });
    assert.ok(model.modelId, "expected non-empty modelId");
  });

  it("multiple subclass instances are independent", () => {
    class LLM_A extends CompletionModel {
      constructor() {
        super({ modelId: "model-a" });
      }
    }
    class LLM_B extends CompletionModel {
      constructor() {
        super({ modelId: "model-b" });
      }
    }
    const a = new LLM_A();
    const b = new LLM_B();
    assert.equal(a.modelId, "model-a");
    assert.equal(b.modelId, "model-b");
    assert.notEqual(a.modelId, b.modelId);
  });
});

// ===========================================================================
// EmbeddingModel + Memory integration (local mode, no API key)
// ===========================================================================

describe("EmbeddingModel subclass cannot be used with Memory", () => {
  it("Memory constructor rejects subclassed EmbeddingModel without inner provider", () => {
    class FakeEmbed extends EmbeddingModel {
      constructor() {
        super({ modelId: "fake", dimensions: 32 });
      }
    }
    const embedder = new FakeEmbed();
    const backend = new InMemoryBackend();

    assert.throws(
      () => new Memory(embedder, backend),
      /concrete EmbeddingModel|not a subclassed instance/
    );
  });
});
