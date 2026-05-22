# frozen_string_literal: true

require "ffi"

module Blazen
  # Configuration for distributed (ring-AllReduce) training. Pass to
  # training verbs (eventually +ModelManager#train_grpo+ /
  # +#train_ppo+) to enable gradient averaging across +world_size+
  # workers connected via gRPC.
  #
  # +rank+ is the 0-indexed rank of this worker; +world_size+ is the
  # total number of workers. +peers+ is the ordered list of
  # +"host:port"+ gRPC endpoints — one entry per rank. +master_addr+ +
  # +master_port+ identify the bootstrap node (typically the host part
  # of +peers[0]+).
  #
  # @example
  #   cfg = Blazen::DistributedConfig.new(
  #     rank: 0,
  #     world_size: 2,
  #     peers: ["host-a:7445", "host-b:7445"],
  #     master_addr: "host-a",
  #     master_port: 7445,
  #   )
  class DistributedConfig
    attr_reader :rank, :world_size, :peers, :master_addr, :master_port

    def initialize(rank:, world_size:, peers:, master_addr:, master_port:)
      @rank = Integer(rank)
      @world_size = Integer(world_size)
      @peers = Array(peers).map(&:to_s)
      @master_addr = master_addr.to_s
      @master_port = Integer(master_port)
      raise ArgumentError, "rank must be < world_size" if @rank >= @world_size
    end

    # Allocate the underlying C struct. Returns an FFI::AutoPointer that
    # frees the struct on garbage collection. Callers pass the pointer to
    # training verbs that accept a distributed configuration.
    def to_native_pointer
      peers_cstr = ::FFI::MemoryPointer.from_string(@peers.join("\n"))
      master_cstr = ::FFI::MemoryPointer.from_string(@master_addr)
      raw = Blazen::FFI.blazen_distributed_config_new(
        @rank,
        @world_size,
        peers_cstr,
        master_cstr,
        @master_port,
      )
      ::FFI::AutoPointer.new(raw, Blazen::FFI.method(:blazen_distributed_config_free))
    end
  end
end
