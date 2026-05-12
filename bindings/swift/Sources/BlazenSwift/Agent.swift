import Foundation
import UniFFIBlazen

/// A configured LLM agent that drives the standard tool-execution loop
/// (completion → execute tool calls → feed results back → repeat).
///
/// Construct with a `CompletionModel`, optional system prompt, the
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
