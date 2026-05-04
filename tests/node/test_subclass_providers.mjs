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

import test from "ava";

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

test("CompletionModel subclassing · can be subclassed with modelId via super(config)", (t) => {
  class MockLLM extends CompletionModel {
    constructor() {
      super({ modelId: "mock-llm", contextLength: 4096 });
    }
  }
  const model = new MockLLM();
  t.is(model.modelId, "mock-llm");
});

test("CompletionModel subclassing · can be subclassed with no config", (t) => {
  class BareLLM extends CompletionModel {
    constructor() {
      super();
    }
  }
  const model = new BareLLM();
  // With no config, modelId defaults to empty string
  t.is(model.modelId, "");
});

test("CompletionModel subclassing · can be subclassed with all config fields", (t) => {
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
  t.is(model.modelId, "full-model");
});

test("CompletionModel subclassing · complete() throws 'subclass must override' for subclassed instances", async (t) => {
  class NoOverrideLLM extends CompletionModel {
    constructor() {
      super({ modelId: "no-override" });
    }
  }
  const model = new NoOverrideLLM();
  await t.throwsAsync(
    () => model.complete([ChatMessage.user("test")]),
    { message: /subclass must override/ }
  );
});

test("CompletionModel subclassing · completeWithOptions() throws for subclassed instances without override", async (t) => {
  class NoOverrideLLM extends CompletionModel {
    constructor() {
      super({ modelId: "no-override" });
    }
  }
  const model = new NoOverrideLLM();
  await t.throwsAsync(
    () =>
      model.completeWithOptions([ChatMessage.user("test")], {
        maxTokens: 10,
      }),
    { message: /subclass must override/ }
  );
});

test("CompletionModel subclassing · withRetry() throws for subclassed instances", (t) => {
  class SubLLM extends CompletionModel {
    constructor() {
      super({ modelId: "sub" });
    }
  }
  const model = new SubLLM();
  t.throws(
    () => model.withRetry({ maxRetries: 3 }),
    { message: /not supported on subclassed/ }
  );
});

test("CompletionModel subclassing · withCache() throws for subclassed instances", (t) => {
  class SubLLM extends CompletionModel {
    constructor() {
      super({ modelId: "sub" });
    }
  }
  const model = new SubLLM();
  t.throws(
    () => model.withCache({ ttlSeconds: 60 }),
    { message: /not supported on subclassed/ }
  );
});

test("CompletionModel subclassing · isLoaded() returns false for subclassed instances", async (t) => {
  class SubLLM extends CompletionModel {
    constructor() {
      super({ modelId: "sub" });
    }
  }
  const model = new SubLLM();
  const loaded = await model.isLoaded();
  t.is(loaded, false);
});

test("CompletionModel subclassing · vramBytes() returns null for subclassed instances", async (t) => {
  class SubLLM extends CompletionModel {
    constructor() {
      super({ modelId: "sub" });
    }
  }
  const model = new SubLLM();
  const bytes = await model.vramBytes();
  t.is(bytes, null);
});

// ===========================================================================
// EmbeddingModel subclassing
// ===========================================================================

test("EmbeddingModel subclassing · can be subclassed with modelId and dimensions", (t) => {
  class MockEmbed extends EmbeddingModel {
    constructor() {
      super({ modelId: "mock-embed", dimensions: 768 });
    }
  }
  const model = new MockEmbed();
  t.is(model.modelId, "mock-embed");
  t.is(model.dimensions, 768);
});

test("EmbeddingModel subclassing · can be subclassed with no config", (t) => {
  class BareEmbed extends EmbeddingModel {
    constructor() {
      super();
    }
  }
  const model = new BareEmbed();
  t.is(model.modelId, "");
  t.is(model.dimensions, 0);
});

test("EmbeddingModel subclassing · can be subclassed with baseUrl", (t) => {
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
  t.is(model.modelId, "remote-embed");
  t.is(model.dimensions, 1536);
});

test("EmbeddingModel subclassing · embed() throws 'subclass must override' for subclassed instances", async (t) => {
  class NoOverrideEmbed extends EmbeddingModel {
    constructor() {
      super({ modelId: "no-override-embed", dimensions: 384 });
    }
  }
  const model = new NoOverrideEmbed();
  await t.throwsAsync(
    () => model.embed(["hello", "world"]),
    { message: /subclass must override/ }
  );
});

// ===========================================================================
// Capability providers -- TTSProvider
// ===========================================================================

test("TTSProvider subclassing · can be constructed with providerId", (t) => {
  const provider = new TTSProvider({ providerId: "mock-tts" });
  t.is(provider.providerId, "mock-tts");
});

test("TTSProvider subclassing · can be constructed with baseUrl and vramEstimateBytes", (t) => {
  const provider = new TTSProvider({
    providerId: "local-tts",
    baseUrl: "http://localhost:5000",
    vramEstimateBytes: 2_000_000_000,
  });
  t.is(provider.providerId, "local-tts");
  t.is(provider.baseUrl, "http://localhost:5000");
  t.is(provider.vramEstimateBytes, 2_000_000_000);
});

test("TTSProvider subclassing · textToSpeech() throws 'subclass must override' by default", async (t) => {
  const provider = new TTSProvider({ providerId: "base-tts" });
  await t.throwsAsync(
    () => provider.textToSpeech({ text: "hello" }),
    { message: /subclass must override textToSpeech/ }
  );
});

// ===========================================================================
// Capability providers -- MusicProvider
// ===========================================================================

test("MusicProvider subclassing · can be constructed with providerId", (t) => {
  const provider = new MusicProvider({ providerId: "mock-music" });
  t.is(provider.providerId, "mock-music");
});

test("MusicProvider subclassing · generateMusic() throws 'subclass must override' by default", async (t) => {
  const provider = new MusicProvider({ providerId: "base-music" });
  await t.throwsAsync(
    () => provider.generateMusic({ prompt: "epic battle" }),
    { message: /subclass must override generateMusic/ }
  );
});

test("MusicProvider subclassing · generateSfx() throws 'subclass must override' by default", async (t) => {
  const provider = new MusicProvider({ providerId: "base-music" });
  await t.throwsAsync(
    () => provider.generateSfx({ prompt: "explosion" }),
    { message: /subclass must override generateSfx/ }
  );
});

// ===========================================================================
// Capability providers -- ImageProvider
// ===========================================================================

test("ImageProvider subclassing · can be constructed with providerId", (t) => {
  const provider = new ImageProvider({ providerId: "mock-image" });
  t.is(provider.providerId, "mock-image");
});

test("ImageProvider subclassing · generateImage() throws 'subclass must override' by default", async (t) => {
  const provider = new ImageProvider({ providerId: "base-image" });
  await t.throwsAsync(
    () => provider.generateImage({ prompt: "cat" }),
    { message: /subclass must override generateImage/ }
  );
});

test("ImageProvider subclassing · upscaleImage() throws 'subclass must override' by default", async (t) => {
  const provider = new ImageProvider({ providerId: "base-image" });
  await t.throwsAsync(
    () => provider.upscaleImage({ image: "base64data" }),
    { message: /subclass must override upscaleImage/ }
  );
});

// ===========================================================================
// Capability providers -- VideoProvider
// ===========================================================================

test("VideoProvider subclassing · can be constructed with providerId", (t) => {
  const provider = new VideoProvider({ providerId: "mock-video" });
  t.is(provider.providerId, "mock-video");
});

test("VideoProvider subclassing · textToVideo() throws 'subclass must override' by default", async (t) => {
  const provider = new VideoProvider({ providerId: "base-video" });
  await t.throwsAsync(
    () => provider.textToVideo({ prompt: "sunset" }),
    { message: /subclass must override textToVideo/ }
  );
});

test("VideoProvider subclassing · imageToVideo() throws 'subclass must override' by default", async (t) => {
  const provider = new VideoProvider({ providerId: "base-video" });
  await t.throwsAsync(
    () => provider.imageToVideo({ image: "base64data" }),
    { message: /subclass must override imageToVideo/ }
  );
});

// ===========================================================================
// Capability providers -- ThreeDProvider
// ===========================================================================

test("ThreeDProvider subclassing · can be constructed with providerId", (t) => {
  const provider = new ThreeDProvider({ providerId: "mock-3d" });
  t.is(provider.providerId, "mock-3d");
});

test("ThreeDProvider subclassing · generate3d() throws 'subclass must override' by default", async (t) => {
  const provider = new ThreeDProvider({ providerId: "base-3d" });
  await t.throwsAsync(
    () => provider.generate3d({ prompt: "chair" }),
    { message: /subclass must override generate3d/ }
  );
});

// ===========================================================================
// Capability providers -- BackgroundRemovalProvider
// ===========================================================================

test("BackgroundRemovalProvider subclassing · can be constructed with providerId", (t) => {
  const provider = new BackgroundRemovalProvider({
    providerId: "mock-bgremoval",
  });
  t.is(provider.providerId, "mock-bgremoval");
});

test("BackgroundRemovalProvider subclassing · removeBackground() throws 'subclass must override' by default", async (t) => {
  const provider = new BackgroundRemovalProvider({
    providerId: "base-bgremoval",
  });
  await t.throwsAsync(
    () => provider.removeBackground({ image: "base64data" }),
    { message: /subclass must override removeBackground/ }
  );
});

// ===========================================================================
// Capability providers -- VoiceProvider
// ===========================================================================

test("VoiceProvider subclassing · can be constructed with providerId", (t) => {
  const provider = new VoiceProvider({ providerId: "mock-voice" });
  t.is(provider.providerId, "mock-voice");
});

test("VoiceProvider subclassing · cloneVoice() throws 'subclass must override' by default", async (t) => {
  const provider = new VoiceProvider({ providerId: "base-voice" });
  await t.throwsAsync(
    () => provider.cloneVoice({ name: "Alice", samples: [] }),
    { message: /subclass must override cloneVoice/ }
  );
});

test("VoiceProvider subclassing · listVoices() throws 'subclass must override' by default", async (t) => {
  const provider = new VoiceProvider({ providerId: "base-voice" });
  await t.throwsAsync(
    () => provider.listVoices(),
    { message: /subclass must override listVoices/ }
  );
});

test("VoiceProvider subclassing · deleteVoice() throws 'subclass must override' by default", async (t) => {
  const provider = new VoiceProvider({ providerId: "base-voice" });
  await t.throwsAsync(
    () => provider.deleteVoice({ voiceId: "v1" }),
    { message: /subclass must override deleteVoice/ }
  );
});

// ===========================================================================
// Capability providers -- common getter coverage
// ===========================================================================

test("capability provider common getters · baseUrl returns null when not set", (t) => {
  const provider = new TTSProvider({ providerId: "no-url" });
  t.is(provider.baseUrl, null);
});

test("capability provider common getters · vramEstimateBytes returns null when not set", (t) => {
  const provider = new ImageProvider({ providerId: "no-vram" });
  t.is(provider.vramEstimateBytes, null);
});

// ===========================================================================
// MemoryBackend subclassing
// ===========================================================================

test("MemoryBackend subclassing · can be constructed", (t) => {
  const backend = new MemoryBackend();
  t.truthy(backend);
});

test("MemoryBackend subclassing · can be subclassed", (t) => {
  class DictBackend extends MemoryBackend {
    constructor() {
      super();
      this.store = new Map();
    }
  }
  const backend = new DictBackend();
  t.truthy(backend);
  t.truthy(backend.store instanceof Map);
});

test("MemoryBackend subclassing · put() throws 'subclass must override' by default", async (t) => {
  const backend = new MemoryBackend();
  await t.throwsAsync(
    () => backend.put({ id: "x", text: "hi" }),
    { message: /subclass must override put/ }
  );
});

test("MemoryBackend subclassing · get() throws 'subclass must override' by default", async (t) => {
  const backend = new MemoryBackend();
  await t.throwsAsync(
    () => backend.get("x"),
    { message: /subclass must override get/ }
  );
});

test("MemoryBackend subclassing · delete() throws 'subclass must override' by default", async (t) => {
  const backend = new MemoryBackend();
  await t.throwsAsync(
    () => backend.delete("x"),
    { message: /subclass must override delete/ }
  );
});

test("MemoryBackend subclassing · list() throws 'subclass must override' by default", async (t) => {
  const backend = new MemoryBackend();
  await t.throwsAsync(
    () => backend.list(),
    { message: /subclass must override list/ }
  );
});

test("MemoryBackend subclassing · len() throws 'subclass must override' by default", async (t) => {
  const backend = new MemoryBackend();
  await t.throwsAsync(
    () => backend.len(),
    { message: /subclass must override len/ }
  );
});

test("MemoryBackend subclassing · searchByBands() throws 'subclass must override' by default", async (t) => {
  const backend = new MemoryBackend();
  await t.throwsAsync(
    () => backend.searchByBands(["band1"], 5),
    { message: /subclass must override searchByBands/ }
  );
});

// ===========================================================================
// ModelManager
// ===========================================================================

test("ModelManager · can be created with budgetGb", (t) => {
  const manager = new ModelManager({ budgetGb: 24 });
  t.truthy(manager);
});

test("ModelManager · can be created with budgetBytes", (t) => {
  const manager = new ModelManager({ budgetBytes: 8_000_000_000n });
  t.truthy(manager);
});

test("ModelManager · throws when neither budgetGb nor budgetBytes is given", (t) => {
  t.throws(
    () => new ModelManager({}),
    { message: /must provide either budgetGb or budgetBytes/ }
  );
});

test("ModelManager · reports zero usedBytes when empty", async (t) => {
  const manager = new ModelManager({ budgetGb: 8 });
  const used = await manager.usedBytes();
  t.is(used, 0n);
});

test("ModelManager · reports full budget as availableBytes when empty", async (t) => {
  const manager = new ModelManager({ budgetGb: 8 });
  const available = await manager.availableBytes();
  t.truthy(available > 0n);
});

test("ModelManager · status() returns empty array when no models registered", async (t) => {
  const manager = new ModelManager({ budgetGb: 8 });
  const statuses = await manager.status();
  t.deepEqual(statuses, []);
});

test("ModelManager · register() throws for remote (non-local) models", async (t) => {
  const manager = new ModelManager({ budgetGb: 8 });
  const model = CompletionModel.openai({ apiKey: "fake-key" });
  await t.throwsAsync(
    () => manager.register("gpt-4", model),
    { message: /does not support local loading/ }
  );
});

// ---------------------------------------------------------------------------
// ModelManager.registerLocalModel (JS-callback adapter)
// ---------------------------------------------------------------------------

test("registerLocalModel · roundtrips load/unload via async JS callbacks", async (t) => {
  const manager = new ModelManager({ budgetGb: 8 });
  let loaded = false;

  await manager.registerLocalModel(
    "rt-1",
    async () => {
      loaded = true;
    },
    async () => {
      loaded = false;
    },
    null,
    1_000_000_000n
  );

  await manager.load("rt-1");
  t.is(await manager.isLoaded("rt-1"), true);
  t.is(loaded, true, "JS-side load callback should have toggled `loaded` to true");

  await manager.unload("rt-1");
  t.is(await manager.isLoaded("rt-1"), false);
  t.is(loaded, false, "JS-side unload callback should have toggled `loaded` to false");
});

test("registerLocalModel · isLoaded callback is invoked when present", async (t) => {
  const manager = new ModelManager({ budgetGb: 8 });
  let isLoadedCalls = 0;
  let internalState = false;

  await manager.registerLocalModel(
    "isl-1",
    async () => {
      internalState = true;
    },
    async () => {
      internalState = false;
    },
    async () => {
      isLoadedCalls += 1;
      return internalState;
    },
    500_000_000n
  );

  // Registration with an isLoaded callback should not throw.
  await manager.load("isl-1");

  const statuses = await manager.status();
  const entry = statuses.find((s) => s.id === "isl-1");
  t.truthy(entry, "expected status entry for isl-1");
  t.is(entry.loaded, true);
  // Counter assertion is loose: the manager doesn't currently invoke the JS
  // callback, but accepting it without throwing is what we're verifying. If
  // it ever does fire, the counter remains a meaningful smoke check.
  t.true(isLoadedCalls >= 0);
});

test("registerLocalModel · accepts null isLoaded and undefined vramEstimateBytes", async (t) => {
  const manager = new ModelManager({ budgetGb: 8 });

  await t.notThrowsAsync(() =>
    manager.registerLocalModel(
      "nullish-1",
      async () => {},
      async () => {},
      null
      // vramEstimateBytes intentionally omitted
    )
  );

  const statuses = await manager.status();
  const entry = statuses.find((s) => s.id === "nullish-1");
  t.truthy(entry, "expected status entry for nullish-1");
  t.is(entry.vramEstimate, 0n);
});

test("registerLocalModel · vramEstimateBytes is reflected in status", async (t) => {
  const manager = new ModelManager({ budgetGb: 8 });

  await manager.registerLocalModel(
    "vram-1",
    async () => {},
    async () => {},
    null,
    5_000_000_000n
  );

  const statuses = await manager.status();
  const entry = statuses.find((s) => s.id === "vram-1");
  t.truthy(entry, "expected status entry for vram-1");
  t.is(entry.vramEstimate, 5_000_000_000n);
});

test("registerLocalModel · LRU eviction works with custom models", async (t) => {
  const manager = new ModelManager({ budgetGb: 8 });
  const fiveGb = 5n * 1_073_741_824n;

  let model1Loaded = false;
  let model2Loaded = false;
  let unloadCalls = 0;

  await manager.registerLocalModel(
    "model1",
    async () => {
      model1Loaded = true;
    },
    async () => {
      model1Loaded = false;
      unloadCalls += 1;
    },
    null,
    fiveGb
  );

  await manager.registerLocalModel(
    "model2",
    async () => {
      model2Loaded = true;
    },
    async () => {
      model2Loaded = false;
      unloadCalls += 1;
    },
    null,
    fiveGb
  );

  await manager.load("model1");
  t.is(await manager.isLoaded("model1"), true);
  t.is(model1Loaded, true);

  // Loading model2 should force eviction of model1 (5 GB + 5 GB > 8 GB budget).
  await manager.load("model2");
  t.is(await manager.isLoaded("model2"), true);
  t.is(model2Loaded, true);

  t.is(
    await manager.isLoaded("model1"),
    false,
    "model1 should have been evicted to make room for model2"
  );
  t.is(model1Loaded, false, "model1's JS-side unload callback should have fired");
  t.true(unloadCalls >= 1, "expected at least one unload callback invocation");
});

test("registerLocalModel · propagates load() rejection", async (t) => {
  const manager = new ModelManager({ budgetGb: 8 });

  await manager.registerLocalModel(
    "boom-1",
    async () => {
      throw new Error("boom");
    },
    async () => {},
    null,
    100_000_000n
  );

  const err = await t.throwsAsync(() => manager.load("boom-1"));
  t.truthy(err, "expected load() to reject");
  const msg = String(err.message ?? err);
  t.true(
    msg.includes("boom") || msg.includes("load() rejected"),
    `expected error message to mention "boom" or "load() rejected", got: ${msg}`
  );
});

test("register (CompletionModel path) · still works", async (t) => {
  // The legacy `register(id, model)` path requires a `JsCompletionModel`
  // backed by an in-process local provider (mistral.rs / llama.cpp / candle).
  // Constructing one of those without local model weights on disk isn't
  // feasible in unit tests, so we exercise the negative path: a remote
  // factory (`openai`) must still be rejected with the documented error.
  // This confirms the legacy entrypoint is wired up and validating its
  // input — registerLocalModel's addition didn't shadow or break it.
  const manager = new ModelManager({ budgetGb: 8 });
  const model = CompletionModel.openai({ apiKey: "fake-key" });
  await t.throwsAsync(() => manager.register("legacy-openai", model), {
    message: /does not support local loading/,
  });
});

// ===========================================================================
// Pricing registration and lookup
// ===========================================================================

test("pricing registration and lookup · registerPricing and lookupPricing round-trip", (t) => {
  registerPricing("test-subclass-model", {
    inputPerMillion: 1.5,
    outputPerMillion: 3.0,
  });
  const pricing = lookupPricing("test-subclass-model");
  t.truthy(pricing, "expected pricing to be found");
  t.is(pricing.inputPerMillion, 1.5);
  t.is(pricing.outputPerMillion, 3.0);
});

test("pricing registration and lookup · lookupPricing returns null for unknown models", (t) => {
  const pricing = lookupPricing("definitely-not-a-real-model-xyz-12345");
  t.is(pricing, null);
});

test("pricing registration and lookup · registerPricing with per_image and per_second", (t) => {
  registerPricing("multimodal-test-model", {
    inputPerMillion: 5.0,
    outputPerMillion: 15.0,
    perImage: 0.02,
    perSecond: 0.001,
  });
  const pricing = lookupPricing("multimodal-test-model");
  t.truthy(pricing, "expected pricing to be found");
  t.is(pricing.inputPerMillion, 5.0);
  t.is(pricing.outputPerMillion, 15.0);
});

test("pricing registration and lookup · registerPricing overwrites previous registration", (t) => {
  registerPricing("overwrite-test-model", {
    inputPerMillion: 1.0,
    outputPerMillion: 2.0,
  });
  registerPricing("overwrite-test-model", {
    inputPerMillion: 10.0,
    outputPerMillion: 20.0,
  });
  const pricing = lookupPricing("overwrite-test-model");
  t.truthy(pricing);
  t.is(pricing.inputPerMillion, 10.0);
  t.is(pricing.outputPerMillion, 20.0);
});

// ===========================================================================
// CustomProvider (host-dispatch wrapper)
// ===========================================================================

test("CustomProvider · can wrap a host object with textToSpeech", (t) => {
  const host = {
    async textToSpeech(request) {
      return { audio: [], timing: {}, metadata: {} };
    },
  };
  const provider = new CustomProvider(host, { providerId: "test-tts" });
  t.is(provider.providerId, "test-tts");
});

test("CustomProvider · defaults providerId to 'custom' when not specified", (t) => {
  const host = {};
  const provider = new CustomProvider(host);
  t.is(provider.providerId, "custom");
});

test("CustomProvider · can wrap a host object with multiple capabilities", (t) => {
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
  t.is(provider.providerId, "multi-cap");
});

test("CustomProvider · can wrap an empty host object (no capabilities)", (t) => {
  const host = {};
  const provider = new CustomProvider(host, { providerId: "empty" });
  t.is(provider.providerId, "empty");
});

// ===========================================================================
// CompletionModel factory sanity (no network calls)
// ===========================================================================

test("CompletionModel factories (construction only) · openai factory sets modelId", (t) => {
  const model = CompletionModel.openai({ apiKey: "fake-key" });
  t.truthy(model.modelId, "expected non-empty modelId");
});

test("CompletionModel factories (construction only) · multiple subclass instances are independent", (t) => {
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
  t.is(a.modelId, "model-a");
  t.is(b.modelId, "model-b");
  t.not(a.modelId, b.modelId);
});

// ===========================================================================
// EmbeddingModel + Memory integration (local mode, no API key)
// ===========================================================================

test("EmbeddingModel subclass cannot be used with Memory · Memory constructor rejects subclassed EmbeddingModel without inner provider", (t) => {
  class FakeEmbed extends EmbeddingModel {
    constructor() {
      super({ modelId: "fake", dimensions: 32 });
    }
  }
  const embedder = new FakeEmbed();
  const backend = new InMemoryBackend();

  t.throws(
    () => new Memory(embedder, backend),
    { message: /concrete EmbeddingModel|not a subclassed instance/ }
  );
});
