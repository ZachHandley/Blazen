# frozen_string_literal: true

module Blazen
  # Distributed peer-to-peer helpers (workflow remoting between Blazen nodes).
  #
  # Requires the native library to have been built with the +distributed+
  # feature.
  module Peer
    module_function

    # Connects a {PeerClient} to a remote Blazen node.
    #
    # @param address [String] +"host:port"+ of the remote node
    # @param client_node_id [String] this client's node ID
    # @return [Blazen::PeerClient]
    def connect(address:, client_node_id:)
      Blazen.translate_errors { Blazen::PeerClient.connect(address, client_node_id) }
    end

    # Creates a new {PeerServer} ready to host workflows for remote clients.
    #
    # @param node_id [String] this server's node ID
    # @return [Blazen::PeerServer]
    def server(node_id:)
      Blazen.translate_errors { Blazen::PeerServer.new(node_id) }
    end
  end
end
