# frozen_string_literal: true

module Blazen
  # Version of the Blazen Ruby gem.
  #
  # Tracks the underlying +blazen-uniffi+ crate version. Use {.version} (defined
  # by the generated UniFFI module) to query the native library's reported
  # version at runtime, which is useful for diagnosing version-skew issues.
  VERSION = "0.0.1"
end
