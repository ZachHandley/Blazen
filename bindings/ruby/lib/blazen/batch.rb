# frozen_string_literal: true

module Blazen
  # Helpers for batch completion.
  #
  # Wraps {Blazen.complete_batch_blocking} with idiomatic error translation
  # and friendlier defaults.
  module Batch
    module_function

    # Runs multiple completion requests against +model+, bounded by
    # +max_concurrency+ in-flight requests.
    #
    # @param model [Blazen::CompletionModel]
    # @param requests [Array<Blazen::CompletionRequest>]
    # @param max_concurrency [Integer] max simultaneous in-flight requests
    # @return [Blazen::BatchResult] aggregated responses + usage totals
    def complete(model, requests, max_concurrency: 4)
      Blazen.translate_errors do
        Blazen.complete_batch_blocking(model, requests, max_concurrency)
      end
    end
  end
end
