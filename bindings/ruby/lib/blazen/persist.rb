# frozen_string_literal: true

module Blazen
  # Persistence (workflow checkpointing) helpers.
  #
  # See {Blazen::CheckpointStore} for the underlying interface. The two
  # built-in backends are:
  #
  # * {redb} — embedded local key-value store backed by a file on disk
  # * {valkey} — Valkey/Redis-compatible server (durable, distributed)
  module Persist
    module_function

    # Opens (or creates) a redb-backed checkpoint store at +path+.
    #
    # @param path [String] filesystem path to the redb database file
    # @return [Blazen::CheckpointStore]
    def redb(path)
      Blazen.translate_errors { Blazen.new_redb_checkpoint_store(path) }
    end

    # Connects to a Valkey/Redis server for checkpoint storage.
    #
    # @param url [String] Valkey/Redis connection URL
    #   (e.g. +"redis://127.0.0.1:6379"+)
    # @param ttl_seconds [Integer, nil] TTL in seconds for stored checkpoints
    # @return [Blazen::CheckpointStore]
    def valkey(url:, ttl_seconds: nil)
      Blazen.translate_errors { Blazen.new_valkey_checkpoint_store(url, ttl_seconds) }
    end
  end
end
