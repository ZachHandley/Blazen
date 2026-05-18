# frozen_string_literal: true

require "spec_helper"
require "json"

# Structural spec for the caller-error preservation path on the Ruby
# binding.
#
# Background
# ----------
#
# Blazen 0.5.4 wires the cabi C ABI to ferry foreign-language exceptions
# raised inside tool handlers back across the FFI boundary as a typed
# `BlazenError::CallerError { name: Option<String>, message: String,
# properties_json: String }`. The Ruby tool-handler trampoline already does
# its half of the job (see `Blazen::Agents.write_caller_error` in
# `lib/blazen/agent.rb`) — on any `StandardError` raised inside the Ruby
# tool handler it captures `exc.class.name`, `exc.message`, and the
# JSON-serialised custom instance variables, then constructs a C-side
# `CallerError` via `blazen_error_from_json`.
#
# What is NOT yet wired up: the Ruby-side decoder. `Blazen.build_error_from_ptr`
# in `lib/blazen/ffi.rb` doesn't yet handle `BLAZEN_ERROR_KIND_CALLER` (kind
# = 19), and `lib/blazen/errors.rb` doesn't yet expose a `Blazen::CallerError`
# class with `name` / `properties_json` accessors.
#
# This spec pins the EXPECTED post-wiring shape:
#
#   1. A Ruby subclass of `Blazen::Error` (`Blazen::CallerError`) with
#      `name` and `properties_json` attributes.
#   2. A tool handler that raises a custom Ruby exception with custom
#      instance variables.
#   3. The agent loop surfaces the failure as a `Blazen::CallerError`
#      where `error.name == "MyCallerError"` and
#      `JSON.parse(error.properties_json) == { "code" => "E_DOMAIN", ... }`.
#
# The spec is `skip`'d at the top of every example until
# `cargo build -p blazen-cabi --release` regenerates `blazen.h` and the
# Ruby-side wrapping code lands.

RSpec.describe "CallerError propagation" do
  before(:all) { Blazen.init }

  # A typed Ruby exception a tool handler might raise to signal a
  # domain-specific failure. Custom instance variables (`@code`, `@detail`)
  # are the kind of structured data that should survive the FFI round-trip
  # via the CallerError `properties_json` blob — `Blazen::Agents.write_caller_error`
  # captures these via the `instance_variables - StandardError.new.instance_variables`
  # difference and JSON-dumps them.
  class MyCallerError < StandardError
    attr_reader :code, :detail

    def initialize(message, code:, detail:)
      super(message)
      @code = code
      @detail = detail
    end
  end

  describe "Blazen::CallerError class (post-wiring)" do
    it "exists as a Blazen::Error subclass with name and properties_json accessors" do
      skip "needs CallerError variant exposed on Blazen — wire " \
           "build_error_from_ptr to BLAZEN_ERROR_KIND_CALLER (19) and " \
           "add Blazen::CallerError to lib/blazen/errors.rb"

      # EXPECTED post-wiring assertions — uncomment once Blazen::CallerError lands.
      #
      # expect(Blazen::CallerError).to be < Blazen::Error
      # err = Blazen::CallerError.new(
      #   "boom",
      #   name: "MyCallerError",
      #   properties_json: '{"code":"E_DOMAIN","detail":"refused"}',
      # )
      # expect(err.name).to eq("MyCallerError")
      # expect(err.properties_json).to eq('{"code":"E_DOMAIN","detail":"refused"}')
    end
  end

  describe "tool-handler exception round-trip" do
    it "surfaces a Ruby StandardError raised in a tool handler as Blazen::CallerError" do
      skip "needs CallerError variant exposed on Blazen — wire " \
           "build_error_from_ptr to BLAZEN_ERROR_KIND_CALLER (19) and " \
           "add Blazen::CallerError to lib/blazen/errors.rb"

      # The Ruby trampoline (Blazen::Agents.write_caller_error) already
      # builds a CallerError handle via blazen_error_from_json on any
      # StandardError raised inside the tool-handler proc — the missing
      # piece is the inverse-direction decoder.
      #
      # EXPECTED post-wiring assertions:
      #
      # model = Blazen::Providers.mock(content: "...")  # regen helper
      # tool = Blazen::Llm.tool(
      #   name: "domain_op",
      #   description: "Always raises a typed caller error.",
      #   parameters: { type: "object", properties: {} },
      # )
      # handler = lambda do |_name, _args|
      #   raise MyCallerError.new("tool refused", code: "E_DOMAIN", detail: "refused")
      # end
      # agent = Blazen::Agents.new(
      #   model: model,
      #   tools: [tool],
      #   tool_handler: handler,
      #   max_iterations: 2,
      # )
      # expect { agent.run_blocking("please call the tool") }
      #   .to raise_error(Blazen::CallerError) do |err|
      #     expect(err.name).to eq("MyCallerError")
      #     payload = JSON.parse(err.properties_json)
      #     expect(payload).to include("code" => "E_DOMAIN", "detail" => "refused")
      #   end
    end
  end

  describe "FFI ERROR_KIND_CALLER constant" do
    it "is exposed on Blazen::FFI as kind 19 (cabi BLAZEN_ERROR_KIND_CALLER)" do
      skip "needs Blazen::FFI::ERROR_KIND_CALLER attached in lib/blazen/ffi.rb"

      # EXPECTED post-wiring: the FFI module exposes the new variant tag
      # alongside the existing ERROR_KIND_* constants.
      #
      # expect(Blazen::FFI::ERROR_KIND_CALLER).to eq(19)
    end

    it "exposes blazen_error_name and blazen_error_properties_json accessors" do
      # These accessors ARE already attached on the Blazen::FFI module —
      # see lib/blazen/ffi.rb lines 263–264. The trampoline at
      # Blazen::Agents.write_caller_error uses blazen_error_from_json to
      # produce caller-error pointers; the decoder side is what's missing.
      #
      # This expectation passes today: it documents the half of the surface
      # that's already in place. Once the spec's other examples activate
      # (i.e. the CallerError class lands), this one stays as a regression
      # guard against accidental detachment.
      expect(Blazen::FFI).to respond_to(:blazen_error_name)
      expect(Blazen::FFI).to respond_to(:blazen_error_properties_json)
      expect(Blazen::FFI).to respond_to(:blazen_error_from_json)
    end
  end
end
