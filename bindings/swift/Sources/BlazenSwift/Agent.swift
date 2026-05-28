import Foundation
import UniFFIBlazen

/// Polymorphic LLM provider handle accepted by `Agent` and `completeBatch`.
/// Wraps any per-engine provider on the Rust side; per-engine concrete
/// classes (`UniFFIBlazen.OpenAiProvider`, `UniFFIBlazen.AnthropicProvider`,
/// ...) do not currently subclass this in Swift — construct an `LlmProvider`
/// via a `CustomProvider` adapter or the engine-specific Rust-side bridges
/// when you need to pass one here.
public typealias LlmProvider = UniFFIBlazen.LlmProvider

/// A configured LLM agent that drives the standard tool-execution loop
/// (completion → execute tool calls → feed results back → repeat).
///
/// Construct with an `LlmProvider`, optional system prompt, the
/// `Tool` definitions the model may call, a `ToolHandler` to execute
/// those calls, and a `maxIterations` budget. Then invoke
/// `run(userInput:)` to drive the loop to completion.
public typealias Agent = UniFFIBlazen.Agent

/// Foreign-callable tool executor invoked by the agent loop. Implement
/// this on any `Sendable` reference type and hand it to the `Agent`
/// constructor along with the `Tool` definitions whose names your
/// `execute(toolName:argumentsJson:)` implementation will dispatch on.
public typealias ToolHandler = UniFFIBlazen.ToolHandler

/// Outcome of an `Agent.run(userInput:)` call.
public typealias AgentResult = UniFFIBlazen.AgentResult
