import Foundation
import UniFFIBlazen

/// A chat completion model handle.
///
/// Build one via the `Providers` factory functions (e.g.
/// `Providers.openAI(apiKey:model:baseURL:)`), then call
/// `complete(_:)` or `completeStream(_:)` to generate responses.
public typealias CompletionModel = UniFFIBlazen.CompletionModel

/// An embedding model handle.
public typealias EmbeddingModel = UniFFIBlazen.EmbeddingModel

/// A single message in a chat conversation. `role` is `"system"`,
/// `"user"`, `"assistant"`, or `"tool"`; `content` is the text payload
/// (empty when the message carries only tool calls or media).
public typealias ChatMessage = UniFFIBlazen.ChatMessage

/// A provider-agnostic chat completion request.
public typealias CompletionRequest = UniFFIBlazen.CompletionRequest

/// The result of a non-streaming chat completion.
public typealias CompletionResponse = UniFFIBlazen.CompletionResponse

/// Response from an embedding model — one vector per input string.
public typealias EmbeddingResponse = UniFFIBlazen.EmbeddingResponse

/// A tool definition the model may invoke during a completion.
public typealias Tool = UniFFIBlazen.Tool

/// A tool invocation requested by the model.
public typealias ToolCall = UniFFIBlazen.ToolCall

/// Multimodal media attached to a `ChatMessage`.
public typealias Media = UniFFIBlazen.Media

/// Token-usage statistics for a completion or embedding request.
public typealias TokenUsage = UniFFIBlazen.TokenUsage

// MARK: - Idiomatic helpers

public extension CompletionModel {
    /// Perform a chat completion. Convenience alias matching the
    /// idiomatic `model.complete(request)` shape preferred over the
    /// underlying argument-labelled `complete(request:)` API.
    func complete(_ request: CompletionRequest) async throws -> CompletionResponse {
        try await complete(request: request)
    }

    /// The model's identifier (e.g. `"gpt-4o"`, `"claude-3-5-sonnet"`).
    /// Re-exposed as a property because callers think of the id as
    /// metadata, not a method.
    var id: String { modelId() }
}

public extension EmbeddingModel {
    /// Embed one or more text strings, returning one vector per input.
    /// Convenience alias matching `model.embed(["hello", "world"])`.
    func embed(_ inputs: [String]) async throws -> EmbeddingResponse {
        try await embed(inputs: inputs)
    }

    /// The model's identifier (e.g. `"text-embedding-3-small"`).
    var id: String { modelId() }

    /// Vector dimensionality. Re-exposed as a property.
    var dimension: UInt32 { dimensions() }
}

// MARK: - Ergonomic constructors for value records

public extension ChatMessage {
    /// Build a `"system"`-role message.
    static func system(_ content: String) -> ChatMessage {
        ChatMessage(
            role: "system",
            content: content,
            mediaParts: [],
            toolCalls: [],
            toolCallId: nil,
            name: nil
        )
    }

    /// Build a `"user"`-role message.
    static func user(_ content: String, media: [Media] = []) -> ChatMessage {
        ChatMessage(
            role: "user",
            content: content,
            mediaParts: media,
            toolCalls: [],
            toolCallId: nil,
            name: nil
        )
    }

    /// Build an `"assistant"`-role message.
    static func assistant(_ content: String, toolCalls: [ToolCall] = []) -> ChatMessage {
        ChatMessage(
            role: "assistant",
            content: content,
            mediaParts: [],
            toolCalls: toolCalls,
            toolCallId: nil,
            name: nil
        )
    }

    /// Build a `"tool"`-role message carrying the result of a previous
    /// tool call. `toolCallId` is the id returned by the model in the
    /// `ToolCall.id` slot.
    static func tool(_ content: String, toolCallId: String) -> ChatMessage {
        ChatMessage(
            role: "tool",
            content: content,
            mediaParts: [],
            toolCalls: [],
            toolCallId: toolCallId,
            name: nil
        )
    }
}

public extension CompletionRequest {
    /// Convenience constructor with `Encodable`-typed messages list and
    /// the rarely-used fields defaulted. Matches the most common call
    /// shape: `CompletionRequest(messages: [...])`.
    init(
        messages: [ChatMessage],
        tools: [Tool] = [],
        temperature: Double? = nil,
        maxTokens: UInt32? = nil,
        topP: Double? = nil,
        model: String? = nil,
        system: String? = nil,
        responseFormatJson: String? = nil
    ) {
        self.init(
            messages: messages,
            tools: tools,
            temperature: temperature,
            maxTokens: maxTokens,
            topP: topP,
            model: model,
            responseFormatJson: responseFormatJson,
            system: system
        )
    }
}
