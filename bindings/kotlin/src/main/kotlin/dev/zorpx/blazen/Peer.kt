package dev.zorpx.blazen

import kotlinx.serialization.Serializable

/**
 * Configuration for a Blazen distributed peer node.
 *
 * Peers expose workflow steps over HTTP (or, when TLS settings are set,
 * HTTPS) and accept inbound requests from other peers. Steps registered
 * here are addressable cluster-wide via [PeerClient].
 */
@Serializable
public data class PeerServerConfig(
    val bindAddress: String,
    val tlsCertPath: String? = null,
    val tlsKeyPath: String? = null,
    val advertisedSteps: List<String> = emptyList(),
)

/**
 * Configuration for the client side of a Blazen distributed peer.
 *
 * `peerAddress` is the base URL of the remote peer (e.g.
 * `https://worker-42.cluster.local:7800`). `tlsCaPath` lets you pin a
 * custom CA bundle for mTLS deployments.
 */
@Serializable
public data class PeerClientConfig(
    val peerAddress: String,
    val tlsCaPath: String? = null,
    val tlsClientCertPath: String? = null,
    val tlsClientKeyPath: String? = null,
)
