import Foundation
import UniFFIBlazen

/// Node-local Blazen peer gRPC server. Construct with `PeerServer(nodeId:)`
/// and start the listener with `serve(listenAddress:)`.
///
/// Only available when the underlying native library was built with the
/// `distributed` feature; otherwise the constructor and `serve` method
/// will not link.
public typealias PeerServer = UniFFIBlazen.PeerServer

/// Client handle for invoking workflows on a remote `PeerServer`.
/// Construct with `PeerClient.connect(address:clientNodeId:)`.
///
/// Only available when the underlying native library was built with the
/// `distributed` feature.
public typealias PeerClient = UniFFIBlazen.PeerClient
