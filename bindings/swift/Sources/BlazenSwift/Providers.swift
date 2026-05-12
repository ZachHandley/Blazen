import Foundation
import UniFFIBlazen

/// Factory namespace for `CompletionModel` and `EmbeddingModel` instances
/// across every Blazen-supported provider.
///
/// Each function returns a fully-constructed model handle ready for
/// `complete(_:)`, `completeStream(_:)`, or `embed(_:)`. Passing the empty
/// string for `apiKey` falls back to the provider's well-known environment
/// variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, ...).
public enum Providers {
    // MARK: Cloud chat-completion providers

    /// Build an OpenAI chat-completion model.
    /// `baseURL` defaults to `https://api.openai.com/v1`; override it to
    /// target any OpenAI-compatible proxy that speaks the official wire
    /// shape.
    public static func openAI(
        apiKey: String,
        model: String? = nil,
        baseURL: String? = nil
    ) throws -> CompletionModel {
        try newOpenaiCompletionModel(apiKey: apiKey, model: model, baseUrl: baseURL)
    }

    /// Build an Anthropic Messages-API chat-completion model.
    public static func anthropic(
        apiKey: String,
        model: String? = nil,
        baseURL: String? = nil
    ) throws -> CompletionModel {
        try newAnthropicCompletionModel(apiKey: apiKey, model: model, baseUrl: baseURL)
    }

    /// Build a Google Gemini chat-completion model.
    public static func gemini(
        apiKey: String,
        model: String? = nil,
        baseURL: String? = nil
    ) throws -> CompletionModel {
        try newGeminiCompletionModel(apiKey: apiKey, model: model, baseUrl: baseURL)
    }

    /// Build an Azure OpenAI chat-completion model.
    /// Azure derives its endpoint from `resourceName` + `deploymentName`
    /// and its model id from `deploymentName`, so neither `model` nor
    /// `baseURL` is exposed here.
    public static func azure(
        apiKey: String,
        resourceName: String,
        deploymentName: String,
        apiVersion: String? = nil
    ) throws -> CompletionModel {
        try newAzureCompletionModel(
            apiKey: apiKey,
            resourceName: resourceName,
            deploymentName: deploymentName,
            apiVersion: apiVersion
        )
    }

    /// Build an AWS Bedrock chat-completion model. `region` selects the
    /// AWS region (e.g. `"us-east-1"`); pass an empty `apiKey` to resolve
    /// it from `AWS_BEARER_TOKEN_BEDROCK`.
    public static func bedrock(
        apiKey: String,
        region: String,
        model: String? = nil,
        baseURL: String? = nil
    ) throws -> CompletionModel {
        try newBedrockCompletionModel(
            apiKey: apiKey,
            region: region,
            model: model,
            baseUrl: baseURL
        )
    }

    /// Build an OpenRouter chat-completion model.
    public static func openRouter(
        apiKey: String,
        model: String? = nil,
        baseURL: String? = nil
    ) throws -> CompletionModel {
        try newOpenrouterCompletionModel(apiKey: apiKey, model: model, baseUrl: baseURL)
    }

    /// Build a Groq chat-completion model.
    public static func groq(
        apiKey: String,
        model: String? = nil,
        baseURL: String? = nil
    ) throws -> CompletionModel {
        try newGroqCompletionModel(apiKey: apiKey, model: model, baseUrl: baseURL)
    }

    /// Build a Together AI chat-completion model.
    public static func together(
        apiKey: String,
        model: String? = nil,
        baseURL: String? = nil
    ) throws -> CompletionModel {
        try newTogetherCompletionModel(apiKey: apiKey, model: model, baseUrl: baseURL)
    }

    /// Build a Mistral chat-completion model.
    public static func mistral(
        apiKey: String,
        model: String? = nil,
        baseURL: String? = nil
    ) throws -> CompletionModel {
        try newMistralCompletionModel(apiKey: apiKey, model: model, baseUrl: baseURL)
    }

    /// Build a DeepSeek chat-completion model.
    public static func deepSeek(
        apiKey: String,
        model: String? = nil,
        baseURL: String? = nil
    ) throws -> CompletionModel {
        try newDeepseekCompletionModel(apiKey: apiKey, model: model, baseUrl: baseURL)
    }

    /// Build a Fireworks AI chat-completion model.
    public static func fireworks(
        apiKey: String,
        model: String? = nil,
        baseURL: String? = nil
    ) throws -> CompletionModel {
        try newFireworksCompletionModel(apiKey: apiKey, model: model, baseUrl: baseURL)
    }

    /// Build a Perplexity chat-completion model.
    public static func perplexity(
        apiKey: String,
        model: String? = nil,
        baseURL: String? = nil
    ) throws -> CompletionModel {
        try newPerplexityCompletionModel(apiKey: apiKey, model: model, baseUrl: baseURL)
    }

    /// Build an xAI (Grok) chat-completion model.
    public static func xAI(
        apiKey: String,
        model: String? = nil,
        baseURL: String? = nil
    ) throws -> CompletionModel {
        try newXaiCompletionModel(apiKey: apiKey, model: model, baseUrl: baseURL)
    }

    /// Build a Cohere chat-completion model.
    public static func cohere(
        apiKey: String,
        model: String? = nil,
        baseURL: String? = nil
    ) throws -> CompletionModel {
        try newCohereCompletionModel(apiKey: apiKey, model: model, baseUrl: baseURL)
    }

    /// Build a fal.ai chat-completion model. `endpoint` selects the
    /// endpoint family (`"openai_chat"`, `"openai_responses"`,
    /// `"openai_embeddings"`, `"openrouter"`, `"any_llm"`); unrecognised
    /// values fall back to `OpenAiChat`. `enterprise` promotes the
    /// endpoint to its SOC2-eligible variant; `autoRouteModality` toggles
    /// automatic routing to a vision/audio/video endpoint when the
    /// request carries media.
    public static func fal(
        apiKey: String,
        model: String? = nil,
        baseURL: String? = nil,
        endpoint: String? = nil,
        enterprise: Bool = false,
        autoRouteModality: Bool = false
    ) throws -> CompletionModel {
        try newFalCompletionModel(
            apiKey: apiKey,
            model: model,
            baseUrl: baseURL,
            endpoint: endpoint,
            enterprise: enterprise,
            autoRouteModality: autoRouteModality
        )
    }

    /// Build a generic OpenAI-compatible chat-completion model. Targets
    /// any service that speaks the official OpenAI Chat Completions wire
    /// format (vLLM, llama-server, LM Studio, ...). Uses
    /// `Authorization: Bearer <apiKey>` auth.
    public static func openAICompatible(
        providerName: String,
        baseURL: String,
        apiKey: String,
        model: String
    ) throws -> CompletionModel {
        try newOpenaiCompatCompletionModel(
            providerName: providerName,
            baseUrl: baseURL,
            apiKey: apiKey,
            model: model
        )
    }

    // MARK: Cloud embedding providers

    /// Build an OpenAI embedding model. Defaults to
    /// `text-embedding-3-small` when `model` is `nil`.
    public static func openAIEmbedding(
        apiKey: String,
        model: String? = nil,
        baseURL: String? = nil
    ) throws -> EmbeddingModel {
        try newOpenaiEmbeddingModel(apiKey: apiKey, model: model, baseUrl: baseURL)
    }

    /// Build a fal.ai embedding model, routed through fal's
    /// OpenAI-compatible embeddings endpoint. `model` defaults to
    /// `"openai/text-embedding-3-small"`; `dimensions` overrides the
    /// produced vector size when the upstream model supports it.
    public static func falEmbedding(
        apiKey: String,
        model: String? = nil,
        dimensions: UInt32? = nil
    ) throws -> EmbeddingModel {
        try newFalEmbeddingModel(apiKey: apiKey, model: model, dimensions: dimensions)
    }

    // MARK: Local backends

    /// Build a local mistral.rs chat-completion model. Available when the
    /// underlying native library was built with the `mistralrs` feature.
    public static func mistralRs(
        modelId: String,
        device: String? = nil,
        quantization: String? = nil,
        contextLength: UInt32? = nil,
        vision: Bool = false
    ) throws -> CompletionModel {
        try newMistralrsCompletionModel(
            modelId: modelId,
            device: device,
            quantization: quantization,
            contextLength: contextLength,
            vision: vision
        )
    }

    /// Build a local llama.cpp chat-completion model. Available when the
    /// underlying native library was built with the `llamacpp` feature.
    public static func llamaCpp(
        modelPath: String,
        device: String? = nil,
        quantization: String? = nil,
        contextLength: UInt32? = nil,
        nGpuLayers: UInt32? = nil
    ) throws -> CompletionModel {
        try newLlamacppCompletionModel(
            modelPath: modelPath,
            device: device,
            quantization: quantization,
            contextLength: contextLength,
            nGpuLayers: nGpuLayers
        )
    }

    /// Build a local candle chat-completion model. Available when the
    /// underlying native library was built with the `candle-llm` feature.
    public static func candle(
        modelId: String,
        device: String? = nil,
        quantization: String? = nil,
        revision: String? = nil,
        contextLength: UInt32? = nil
    ) throws -> CompletionModel {
        try newCandleCompletionModel(
            modelId: modelId,
            device: device,
            quantization: quantization,
            revision: revision,
            contextLength: contextLength
        )
    }

    /// Build a local fastembed (ONNX Runtime) embedding model. Available
    /// when the underlying native library was built with the `embed`
    /// feature.
    public static func fastEmbed(
        modelName: String? = nil,
        maxBatchSize: UInt32? = nil,
        showDownloadProgress: Bool? = nil
    ) throws -> EmbeddingModel {
        try newFastembedEmbeddingModel(
            modelName: modelName,
            maxBatchSize: maxBatchSize,
            showDownloadProgress: showDownloadProgress
        )
    }

    /// Build a local candle text-embedding model. Available when the
    /// underlying native library was built with the `candle-embed`
    /// feature.
    public static func candleEmbedding(
        modelId: String? = nil,
        device: String? = nil,
        revision: String? = nil
    ) throws -> EmbeddingModel {
        try newCandleEmbeddingModel(modelId: modelId, device: device, revision: revision)
    }

    /// Build a local tract (pure-Rust ONNX) embedding model. Drop-in
    /// replacement for `fastEmbed(...)` when prebuilt ONNX Runtime
    /// binaries can't link. Available when the underlying native library
    /// was built with the `tract` feature.
    public static func tractEmbedding(
        modelName: String? = nil,
        maxBatchSize: UInt32? = nil,
        showDownloadProgress: Bool? = nil
    ) throws -> EmbeddingModel {
        try newTractEmbeddingModel(
            modelName: modelName,
            maxBatchSize: maxBatchSize,
            showDownloadProgress: showDownloadProgress
        )
    }
}
