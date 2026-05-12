# frozen_string_literal: true

# Minimal "hello world" demonstrating the gem entrypoint, version reporting,
# and the chat-message builder.
#
# Running it end-to-end requires the native lib (+libblazen_uniffi.so+) to be
# discoverable — either bundled in +ext/blazen/+ (the default packaging) or
# pointed at via the +BLAZEN_UNIFFI_LIB_PATH+ environment variable.
#
# Workflow execution with Ruby-defined step handlers is not yet supported
# because the upstream uniffi-bindgen Ruby template doesn't generate
# foreign-callback scaffolding (see lib/blazen/workflow.rb for details).
# This example sticks to surface-level calls that don't need callbacks.

require "blazen"

Blazen.init

puts "Blazen native lib version: #{Blazen.version}"
puts "Blazen gem version:        #{Blazen::VERSION}"

req = Blazen::Llm.completion_request(
  messages: [
    Blazen::Llm.system("You answer in one short sentence."),
    Blazen::Llm.user("What is the airspeed velocity of an unladen swallow?"),
  ],
  temperature: 0.0,
  max_tokens: 64,
)
puts "Constructed CompletionRequest with #{req.messages.length} messages."

# Actually calling out to a provider requires an API key:
#
#   model = Blazen::Providers.openai(api_key: ENV.fetch("OPENAI_API_KEY"))
#   resp = model.complete_blocking(req)
#   puts resp.content

Blazen.shutdown
