# frozen_string_literal: true

require "json"

module Blazen
  # Tool-calling LLM-agent loop, with first-class support for
  # Ruby-implemented tool handlers via the cabi
  # +BlazenToolHandlerVTable+ trampoline.
  #
  # Use {Blazen::Agents.new} to construct an {Agent}: supply a backing
  # provider ({Blazen::LlmProvider} or a per-engine class like
  # {Blazen::OpenAiProvider} that responds to +#as_llm_provider+), a list
  # of {Blazen::Llm::Tool} declarations, a callable +tool_handler+, and
  # (optionally) a +system_prompt+ / +max_iterations+. The handler is invoked by the Rust agent loop every time
  # the model emits a tool call and must return either a +String+ (forwarded
  # verbatim) or a JSON-serializable Ruby value (encoded via +JSON.dump+).
  #
  # @example Build an agent with a Ruby-defined tool handler
  #   tool = Blazen::Llm.tool(
  #     name: "now",
  #     description: "Returns the current UTC timestamp.",
  #     parameters: { type: "object", properties: {} },
  #   )
  #   handler = ->(_name, _args) { { utc: Time.now.utc.iso8601 } }
  #   agent = Blazen::Agents.new(
  #     model: model,
  #     tools: [tool],
  #     tool_handler: handler,
  #     system_prompt: "Reply concisely.",
  #     max_iterations: 4,
  #   )
  #   result = agent.run_blocking("What time is it?")
  #   puts result.final_message
  #
  # @note Allocator compatibility
  #   The tool-handler trampoline writes its result string back to Rust via
  #   {::FFI::MemoryPointer.from_string}. On the Rust side
  #   ({Blazen::FFI.blazen_string_free}) the buffer is freed via
  #   +CString::from_raw+, which calls the system allocator. On Linux / macOS
  #   both Ruby's {::FFI} gem and Rust's +CString+ allocate through libc's
  #   +malloc+, so freeing one with the other is sound. If a future build
  #   targets a non-glibc / non-libSystem platform where Ruby and Rust use
  #   distinct allocators, this code will need to switch to a
  #   +blazen_string_alloc+ helper on the cabi side (deferred to phase R9).
  module Agents
    # ---------------------------------------------------------------------
    # Tool-handler callback registry
    # ---------------------------------------------------------------------
    #
    # Each Agent registers its Ruby tool-handler proc in a process-wide
    # registry keyed by a small integer ID. The cabi receives the ID as the
    # vtable's opaque +user_data+ pointer (cast through +Pointer.new+) and
    # passes it back on every +execute+ / +drop_user_data+ invocation. This
    # indirection is mandatory because the Ruby +ffi+ gem GC-tracks +Proc+
    # objects: a raw +Proc#object_id+ won't survive the round-trip through
    # +*mut c_void+. The registry holds a strong reference until
    # +drop_user_data+ fires (which happens when the inner +CToolHandler+
    # wrapper drops on the Rust side).
    @tool_registry = {}
    @tool_registry_mutex = Mutex.new
    @next_tool_id = 0

    # @api private
    # @return [Integer] new registry ID
    def self.register_tool_handler(handler)
      @tool_registry_mutex.synchronize do
        @next_tool_id += 1
        id = @next_tool_id
        @tool_registry[id] = handler
        id
      end
    end

    # @api private
    def self.unregister_tool_handler(id)
      @tool_registry_mutex.synchronize { @tool_registry.delete(id) }
    end

    # @api private
    # @return [#call, nil]
    def self.lookup_tool_handler(id)
      @tool_registry_mutex.synchronize { @tool_registry[id] }
    end

    # +execute+ thunk for the +BlazenToolHandlerVTable+. Looks up the
    # Ruby-side tool handler by its registry ID (passed back as +user_data+),
    # parses the JSON arguments, invokes the handler, and writes the result
    # back to Rust as a heap-allocated NUL-terminated C string.
    #
    # The handler is invoked with +(tool_name, args_hash)+. Its return value
    # is coerced to a JSON string: +String+ values are forwarded verbatim
    # (the handler is responsible for them being well-formed JSON for the
    # model), every other value goes through +JSON.dump+.
    #
    # On any Ruby exception we construct a +BlazenError::CallerError+ via the
    # C ABI's +blazen_error_from_json+, write the handle to +out_err+, and
    # return +-1+. The error carries the original exception's +class.name+,
    # +message+, and any custom instance variables serialised as
    # +properties_json+ — so foreign callers see +error.name == "MyError"+
    # and can JSON-decode +error.properties_json+ to recover the payload.
    # Full instanceof preservation isn't possible through the C ABI (FFI
    # mechanism is genuinely missing); structural preservation is the
    # documented binding-parity exception for UniFFI/cabi bindings.
    EXECUTE_FN = ::FFI::Function.new(
      :int32,
      %i[pointer pointer pointer pointer pointer],
      proc do |user_data_ptr, tool_name_ptr, arguments_json_ptr, out_result_json, out_err|
        id = user_data_ptr.address
        handler = lookup_tool_handler(id)

        if handler.nil?
          warn "[blazen] tool handler not found for id=#{id}"
          next(-1)
        end

        begin
          tool_name = tool_name_ptr.null? ? "" : tool_name_ptr.read_string
          arguments_json = arguments_json_ptr.null? ? "" : arguments_json_ptr.read_string
          args = arguments_json.empty? ? {} : JSON.parse(arguments_json)

          result = handler.call(tool_name, args)
          result_str =
            case result
            when String then result
            else JSON.dump(result)
            end

          # Hand a NUL-terminated buffer back to Rust. The Rust trampoline
          # reclaims it via +CString::from_raw+ (i.e. the system allocator),
          # so we MUST disable AutoRelease here — otherwise Ruby would free
          # the same buffer on GC and Rust would double-free it.
          ptr = ::FFI::MemoryPointer.from_string(result_str)
          ptr.autorelease = false
          out_result_json.write_pointer(ptr)
          0
        rescue StandardError => e
          write_caller_error(out_err, e)
          -1
        end
      end,
      blocking: true,
    )

    # Builds a +BlazenError::CallerError+ via the C ABI and writes the
    # resulting handle into the +out_err+ slot supplied by the Rust
    # trampoline. Captures +e.class.name+ as +name+, +e.message+ as
    # +message+, and JSON-marshals any custom instance variables defined
    # on +e+ (i.e. those not present on +StandardError+) into
    # +properties_json+. Non-JSON-serialisable values fall back to their
    # +inspect+ string.
    def self.write_caller_error(out_err, exc)
      return if out_err.nil? || out_err.null?

      props = {}
      base_ivars = StandardError.new.instance_variables
      (exc.instance_variables - base_ivars).each do |ivar|
        key = ivar.to_s.delete_prefix("@")
        value = exc.instance_variable_get(ivar)
        props[key] =
          begin
            JSON.parse(JSON.dump(value))
          rescue StandardError
            value.inspect
          end
      end

      payload = {
        "kind" => "CallerError",
        "name" => exc.class.name,
        "message" => exc.message,
        "properties_json" => JSON.dump(props),
      }
      json_str = JSON.dump(payload)
      err_ptr = Blazen::FFI.blazen_error_from_json(::FFI::MemoryPointer.from_string(json_str))
      out_err.write_pointer(err_ptr) unless err_ptr.null?
    rescue StandardError => sub_err
      warn "[blazen] failed to construct CallerError handle: #{sub_err.class}: #{sub_err.message}"
    end

    # +drop_user_data+ thunk for the +BlazenToolHandlerVTable+. Releases the
    # registry slot when the inner +CToolHandler+ drops on the Rust side
    # (i.e. when the last reference to the underlying +Agent+ goes away).
    DROP_TOOL_USER_DATA_FN = ::FFI::Function.new(
      :void,
      [:pointer],
      proc do |user_data_ptr|
        unregister_tool_handler(user_data_ptr.address)
      end,
    )

    # ---------------------------------------------------------------------
    # Public types
    # ---------------------------------------------------------------------

    # Wraps a +BlazenAgent *+ — the orchestrator that drives the
    # tool-calling loop until the model emits a final assistant message
    # (or +max_iterations+ is exhausted).
    #
    # Construct via {Blazen::Agents.new}; user code shouldn't call +.new+
    # directly.
    class Agent
      # @api private
      # @param raw_ptr [::FFI::Pointer] caller-owned +BlazenAgent *+
      def initialize(raw_ptr)
        if raw_ptr.nil? || raw_ptr.null?
          raise Blazen::InternalError, "Agent: native constructor returned null"
        end

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_agent_free))
      end

      # @return [::FFI::AutoPointer] the underlying native handle
      attr_reader :ptr

      # Runs the agent loop asynchronously, returning an {AgentResult} when
      # the future resolves. Composes with +Fiber.scheduler+ when one is
      # active (see {Blazen::FFI.await_future}).
      #
      # @param user_input [String] the initial user message
      # @return [AgentResult]
      def run(user_input)
        fut =
          Blazen::FFI.with_cstring(user_input) do |i|
            Blazen::FFI.blazen_agent_run(@ptr, i)
          end
        raise Blazen::InternalError, "blazen_agent_run returned null" if fut.nil? || fut.null?

        Blazen::FFI.await_future(fut) do |f|
          out_result = ::FFI::MemoryPointer.new(:pointer)
          out_err = ::FFI::MemoryPointer.new(:pointer)
          Blazen::FFI.blazen_future_take_agent_result(f, out_result, out_err)
          Blazen::FFI.check_error!(out_err)
          AgentResult.new(out_result.read_pointer)
        end
      end

      # Runs the agent loop synchronously, blocking the calling thread on
      # the cabi tokio runtime.
      #
      # @param user_input [String] the initial user message
      # @return [AgentResult]
      def run_blocking(user_input)
        out_result = ::FFI::MemoryPointer.new(:pointer)
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(user_input) do |i|
          Blazen::FFI.blazen_agent_run_blocking(@ptr, i, out_result, out_err)
        end
        Blazen::FFI.check_error!(out_err)
        AgentResult.new(out_result.read_pointer)
      end
    end

    # The terminal payload returned by {Agent#run} / {Agent#run_blocking}.
    # Wraps +BlazenAgentResult *+.
    class AgentResult
      # @api private
      # @param raw_ptr [::FFI::Pointer] caller-owned +BlazenAgentResult *+
      def initialize(raw_ptr)
        if raw_ptr.nil? || raw_ptr.null?
          raise Blazen::InternalError, "AgentResult: native constructor returned null"
        end

        @ptr = ::FFI::AutoPointer.new(raw_ptr, Blazen::FFI.method(:blazen_agent_result_free))
      end

      # @return [::FFI::AutoPointer]
      attr_reader :ptr

      # @return [String, nil] the final assistant message
      def final_message
        Blazen::FFI.consume_cstring(Blazen::FFI.blazen_agent_result_final_message(@ptr))
      end

      # @return [Integer] number of (model-call, tool-execution) iterations
      def iterations
        Blazen::FFI.blazen_agent_result_iterations(@ptr)
      end

      # @return [Integer] total number of tool invocations across the run
      def tool_call_count
        Blazen::FFI.blazen_agent_result_tool_call_count(@ptr)
      end

      # @return [Float] aggregate cost across every model call (USD)
      def total_cost_usd
        Blazen::FFI.blazen_agent_result_total_cost_usd(@ptr)
      end

      # @return [Blazen::Llm::TokenUsage, nil] aggregate token counts
      def total_usage
        raw = Blazen::FFI.blazen_agent_result_total_usage(@ptr)
        return nil if raw.nil? || raw.null?

        Blazen::Llm::TokenUsage.from_raw(raw)
      end
    end

    module_function

    # Constructs a new {Agent}.
    #
    # The +tool_handler+ is invoked from the Rust agent loop every time the
    # model emits a tool call. It receives +(tool_name, args_hash)+ — the
    # +args_hash+ is the JSON-decoded arguments object the model produced —
    # and must return either a +String+ (forwarded verbatim to the model
    # as the tool result) or a JSON-serializable Ruby value (encoded via
    # +JSON.dump+).
    #
    # Each +Blazen::Llm::Tool+ passed via +tools+ is consumed by the call
    # (the cabi +blazen_agent_new+ takes ownership of every tool handle);
    # subsequent reads through the original Ruby wrapper raise
    # {Blazen::InternalError}.
    #
    # @param provider [Blazen::LlmProvider, #as_llm_provider] backing
    #   completion provider — either a polymorphic {Blazen::LlmProvider}
    #   (typically produced by per-engine +#as_llm_provider+) OR any
    #   per-engine class that responds to +:as_llm_provider+
    #   (auto-converted before hand-off to the cabi).
    # @param tool_handler [#call] callable taking +(tool_name, args_hash)+
    # @param tools [Array<Blazen::Llm::Tool>] declared tool list shown to the
    #   model (consumed)
    # @param system_prompt [String, nil] optional system prompt
    # @param max_iterations [Integer] safety cap on loop iterations
    #   (default: 16)
    # @return [Agent]
    def new(provider: nil, tool_handler:, tools: [], system_prompt: nil,
            max_iterations: 16, model: nil)
      # Back-compat: the previous signature was +model:+; accept either.
      provider ||= model
      raise ArgumentError, "provider: is required" if provider.nil?

      llm_provider =
        if provider.is_a?(Blazen::LlmProvider)
          provider.as_llm_provider
        elsif provider.respond_to?(:as_llm_provider)
          provider.as_llm_provider
        else
          raise ArgumentError,
                "provider must be a Blazen::LlmProvider or respond to #as_llm_provider"
        end
      unless llm_provider.is_a?(Blazen::LlmProvider)
        raise ArgumentError,
              "#as_llm_provider must return a Blazen::LlmProvider " \
                "(got #{llm_provider.class})"
      end

      unless tool_handler.respond_to?(:call)
        raise ArgumentError, "tool_handler must respond to #call(tool_name, args_hash)"
      end

      tool_array = Array(tools)
      tool_array.each do |tool|
        unless tool.is_a?(Blazen::Llm::Tool)
          raise ArgumentError, "tools entries must be Blazen::Llm::Tool"
        end
      end

      tool_id = register_tool_handler(tool_handler)
      registered = true

      begin
        vtable = Blazen::FFI::BlazenToolHandlerVTable.new
        vtable[:user_data] = ::FFI::Pointer.new(tool_id)
        vtable[:drop_user_data] = DROP_TOOL_USER_DATA_FN
        vtable[:execute] = EXECUTE_FN

        # Consume each Tool wrapper into a bare pointer; the cabi takes
        # ownership of every element in the array.
        tool_ptrs = tool_array.map(&:consume!)
        tools_arr =
          if tool_ptrs.empty?
            ::FFI::Pointer::NULL
          else
            buf = ::FFI::MemoryPointer.new(:pointer, tool_ptrs.length)
            tool_ptrs.each_with_index { |p, i| buf[i].write_pointer(p) }
            buf
          end

        out_agent = ::FFI::MemoryPointer.new(:pointer)
        out_err = ::FFI::MemoryPointer.new(:pointer)
        Blazen::FFI.with_cstring(system_prompt) do |sp|
          Blazen::FFI.blazen_agent_new(
            llm_provider.ptr,
            sp,
            tools_arr,
            tool_ptrs.length,
            vtable,
            max_iterations.to_i,
            out_agent,
            out_err,
          )
        end

        # On any error, +blazen_agent_new+ has already invoked
        # +drop_user_data+ on its end, which unregistered our handler. Mark
        # +registered+ accordingly so the +ensure+ arm doesn't double-drop.
        registered = false
        Blazen::FFI.check_error!(out_err)

        Agent.new(out_agent.read_pointer)
      ensure
        # Defensive cleanup: if we raised BEFORE handing the vtable off to
        # the cabi (e.g. ArgumentError above, or a synchronous Ruby
        # exception while building +tools_arr+), the registry entry would
        # leak. Unregister it here. After a successful +blazen_agent_new+
        # call the cabi owns the lifetime of the registration via
        # +drop_user_data+, so we leave it alone (+registered+ is false).
        unregister_tool_handler(tool_id) if registered
      end
    end
  end
end
