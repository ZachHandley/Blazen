//! Ring topology bootstrap helpers.
//!
//! A ring of `N` workers is represented by:
//!
//! - the local worker's `rank` in `0..N`,
//! - the total `world_size` `N`,
//! - the gRPC endpoints of every peer (so this worker can find both its
//!   `next` and `prev` neighbors without an extra lookup hop).
//!
//! `next` is `(rank + 1) mod N`; `prev` is `(rank + N - 1) mod N`.

use std::fmt;

use crate::error::BlazenTrainError;

/// Topology of one worker inside an N-worker ring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingTopology {
    /// 0-indexed rank of this worker.
    pub rank: usize,
    /// Total number of workers in the ring.
    pub world_size: usize,
    /// gRPC endpoint of every worker in the ring, indexed by rank.
    pub peers: Vec<String>,
}

impl RingTopology {
    /// Build a topology from the local rank and the full peer list.
    ///
    /// # Errors
    /// - [`BlazenTrainError::InvalidConfig`] if `rank >= peers.len()` or
    ///   `peers.is_empty()`.
    pub fn new(rank: usize, peers: Vec<String>) -> Result<Self, BlazenTrainError> {
        if peers.is_empty() {
            return Err(BlazenTrainError::InvalidConfig(
                "RingTopology requires at least one peer endpoint".to_string(),
            ));
        }
        if rank >= peers.len() {
            return Err(BlazenTrainError::InvalidConfig(format!(
                "RingTopology rank {rank} is out of range for world_size {}",
                peers.len()
            )));
        }
        let world_size = peers.len();
        Ok(Self {
            rank,
            world_size,
            peers,
        })
    }

    /// Endpoint of the `next` peer (the worker this rank sends to).
    #[must_use]
    pub fn next_endpoint(&self) -> &str {
        let n = self.world_size;
        &self.peers[(self.rank + 1) % n]
    }

    /// Endpoint of the `prev` peer (the worker this rank receives from).
    #[must_use]
    pub fn prev_endpoint(&self) -> &str {
        let n = self.world_size;
        &self.peers[(self.rank + n - 1) % n]
    }

    /// Rank of the `next` peer.
    #[must_use]
    pub fn next_rank(&self) -> usize {
        (self.rank + 1) % self.world_size
    }

    /// Rank of the `prev` peer.
    #[must_use]
    pub fn prev_rank(&self) -> usize {
        (self.rank + self.world_size - 1) % self.world_size
    }

    /// `true` when `world_size > 1` — the trainer should AllReduce.
    #[must_use]
    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
    }
}

impl fmt::Display for RingTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RingTopology(rank={}, world_size={}, next={}, prev={})",
            self.rank,
            self.world_size,
            self.next_endpoint(),
            self.prev_endpoint(),
        )
    }
}

/// Distributed-training configuration carried alongside [`crate::TrainConfig`].
///
/// `peers` is the full ordered list of `"host:port"` endpoints; index `i`
/// is rank `i`. `master_addr` / `master_port` identify the bootstrap /
/// rendezvous node (typically `peers[0]`) and are persisted on the
/// config for parity with PyTorch's `torchrun` env vars.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RingConfig {
    /// 0-indexed rank of this worker.
    pub rank: usize,
    /// Total number of workers.
    pub world_size: usize,
    /// Full peer endpoint list, indexed by rank.
    pub peers: Vec<String>,
    /// Bootstrap address (typically the rank-0 host).
    pub master_addr: String,
    /// Bootstrap port.
    pub master_port: u16,
}

impl RingConfig {
    /// Construct a config from a comma-separated peer list.
    ///
    /// Mirrors the bootstrap format used by `torchrun --master_addr ... \
    /// --master_port ... --node_rank ... --nnodes ...` but collapsed
    /// into one string so it can travel across the bindings without a
    /// dedicated struct.
    ///
    /// `peer_list` looks like `"host1:50051,host2:50051,host3:50051"`.
    /// `world_size` is inferred from the comma count.
    ///
    /// # Errors
    /// - [`BlazenTrainError::InvalidConfig`] if the list is empty,
    ///   contains a malformed `"host:port"` entry, or `rank` is out of
    ///   range.
    pub fn from_peer_list(rank: usize, peer_list: &str) -> Result<Self, BlazenTrainError> {
        let peers = parse_peer_list(peer_list)?;
        if rank >= peers.len() {
            return Err(BlazenTrainError::InvalidConfig(format!(
                "rank {rank} is out of range for world_size {}",
                peers.len()
            )));
        }
        let world_size = peers.len();
        let master = &peers[0];
        let (master_addr, master_port) = split_host_port(master)?;
        Ok(Self {
            rank,
            world_size,
            peers,
            master_addr,
            master_port,
        })
    }

    /// Materialize the matching [`RingTopology`] for this worker.
    ///
    /// # Errors
    /// Forwards any error from [`RingTopology::new`].
    pub fn topology(&self) -> Result<RingTopology, BlazenTrainError> {
        RingTopology::new(self.rank, self.peers.clone())
    }

    /// Convenience flag — `true` when `world_size > 1`.
    #[must_use]
    pub fn is_distributed(&self) -> bool {
        self.world_size > 1
    }
}

/// Parse a comma-separated `"host:port,host:port,..."` peer list.
///
/// Whitespace around entries is trimmed. Empty entries are rejected.
///
/// # Errors
/// - [`BlazenTrainError::InvalidConfig`] if the list is empty after
///   trimming or contains an entry without a `:`.
pub fn parse_peer_list(s: &str) -> Result<Vec<String>, BlazenTrainError> {
    let peers: Vec<String> = s
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(ToString::to_string)
        .collect();
    if peers.is_empty() {
        return Err(BlazenTrainError::InvalidConfig(
            "peer list is empty".to_string(),
        ));
    }
    for p in &peers {
        if !p.contains(':') {
            return Err(BlazenTrainError::InvalidConfig(format!(
                "peer entry {p:?} missing :port"
            )));
        }
    }
    Ok(peers)
}

fn split_host_port(s: &str) -> Result<(String, u16), BlazenTrainError> {
    let (host, port) = s.rsplit_once(':').ok_or_else(|| {
        BlazenTrainError::InvalidConfig(format!("master endpoint {s:?} missing :port"))
    })?;
    let port_num: u16 = port
        .parse()
        .map_err(|e| BlazenTrainError::InvalidConfig(format!("invalid port in {s:?}: {e}")))?;
    Ok((host.to_string(), port_num))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_topology_bootstrap_from_rank_list_string() {
        let cfg =
            RingConfig::from_peer_list(1, "host1:50051, host2:50051 , host3:50051 ,host4:50051")
                .expect("parse");
        assert_eq!(cfg.rank, 1);
        assert_eq!(cfg.world_size, 4);
        assert_eq!(cfg.peers.len(), 4);
        assert_eq!(cfg.peers[0], "host1:50051");
        assert_eq!(cfg.peers[3], "host4:50051");
        assert_eq!(cfg.master_addr, "host1");
        assert_eq!(cfg.master_port, 50051);
        assert!(cfg.is_distributed());

        let topo = cfg.topology().expect("topology");
        assert_eq!(topo.next_endpoint(), "host3:50051");
        assert_eq!(topo.prev_endpoint(), "host1:50051");
        assert_eq!(topo.next_rank(), 2);
        assert_eq!(topo.prev_rank(), 0);
    }

    #[test]
    fn ring_topology_wraps_at_boundaries() {
        let peers = vec!["a:1".to_string(), "b:2".to_string(), "c:3".to_string()];
        let topo = RingTopology::new(0, peers.clone()).expect("new");
        assert_eq!(topo.next_endpoint(), "b:2");
        assert_eq!(topo.prev_endpoint(), "c:3");

        let topo_last = RingTopology::new(2, peers).expect("new");
        assert_eq!(topo_last.next_endpoint(), "a:1");
        assert_eq!(topo_last.prev_endpoint(), "b:2");
    }

    #[test]
    fn ring_topology_rejects_out_of_range_rank() {
        let peers = vec!["a:1".to_string()];
        let err = RingTopology::new(5, peers).unwrap_err();
        assert!(matches!(err, BlazenTrainError::InvalidConfig(_)));
    }

    #[test]
    fn ring_topology_rejects_empty_peers() {
        let err = RingTopology::new(0, vec![]).unwrap_err();
        assert!(matches!(err, BlazenTrainError::InvalidConfig(_)));
    }

    #[test]
    fn parse_peer_list_rejects_missing_port() {
        let err = parse_peer_list("host1,host2:50051").unwrap_err();
        assert!(matches!(err, BlazenTrainError::InvalidConfig(_)));
    }

    #[test]
    fn parse_peer_list_rejects_empty_string() {
        let err = parse_peer_list("   ").unwrap_err();
        assert!(matches!(err, BlazenTrainError::InvalidConfig(_)));
    }

    #[test]
    fn ring_topology_world_one_is_not_distributed() {
        let topo = RingTopology::new(0, vec!["solo:7".to_string()]).expect("new");
        assert!(!topo.is_distributed());
    }
}
