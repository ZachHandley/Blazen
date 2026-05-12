# frozen_string_literal: true

require_relative "lib/blazen/version"

Gem::Specification.new do |spec|
  spec.name          = "blazen"
  spec.version       = Blazen::VERSION
  spec.authors       = ["Zach Handley"]
  spec.email         = ["zach@zorpxinc.com"]

  spec.summary       = "Ruby bindings for Blazen, an LLM-orchestration framework"
  spec.description   = <<~DESC
    Ruby bindings for Blazen — a Rust LLM-orchestration framework with first-class
    support for workflows, pipelines, agents, batch and streaming completions,
    embeddings, image / audio compute, distributed peer execution, and durable
    checkpointing. Glue is generated via Mozilla UniFFI and wrapped with an
    idiomatic Ruby API.
  DESC
  spec.homepage      = "https://github.com/zachhandley/Blazen"
  spec.license       = "MPL-2.0"
  spec.required_ruby_version = ">= 3.0.0"

  spec.metadata["homepage_uri"]    = spec.homepage
  spec.metadata["source_code_uri"] = spec.homepage
  spec.metadata["bug_tracker_uri"] = "#{spec.homepage}/issues"

  spec.files = Dir[
    "lib/**/*.rb",
    "ext/blazen/libblazen_uniffi.so",
    "ext/blazen/libblazen_uniffi.dylib",
    "ext/blazen/libblazen_uniffi.dll",
    "blazen.gemspec",
    "README.md",
    "LICENSE",
  ].select { |f| File.exist?(f) }

  spec.require_paths = ["lib"]

  spec.add_dependency "ffi", "~> 1.16"

  spec.add_development_dependency "rake", "~> 13.0"
  spec.add_development_dependency "rspec", "~> 3.13"
end
