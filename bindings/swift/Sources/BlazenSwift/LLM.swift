import Foundation
import UniFFIBlazen

/// A single message in a chat conversation. `role` is `"system"`,
/// `"user"`, `"assistant"`, or `"tool"`; `content` is the text payload
/// (empty when the message carries only tool calls or media).
public typealias ChatMessage = UniFFIBlazen.ChatMessage

/// A provider-agnostic chat completion request.
public typealias ModelRequest = UniFFIBlazen.ModelRequest

/// The result of a non-streaming chat completion.
public typealias ModelResponse = UniFFIBlazen.ModelResponse

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

public extension ModelRequest {
    /// Convenience constructor with `Encodable`-typed messages list and
    /// the rarely-used fields defaulted. Matches the most common call
    /// shape: `ModelRequest(messages: [...])`.
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
