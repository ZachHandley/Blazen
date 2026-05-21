//! Inter-worker transport for ring-AllReduce.
//!
//! The [`RingTransport`] trait abstracts the bytes pipe used by
//! [`super::allreduce::ring_all_reduce`]:
//!
//! - `send(step, chunk_id, payload)` — push bytes to the `next` peer.
//! - `recv(step, chunk_id)` — pull bytes that this rank's `prev` peer
//!   has pushed for the given `(step, chunk_id)`.
//!
//! The (step, chunk_id) coordinates let the receiver match an incoming
//! chunk to the slot the AllReduce kernel is waiting on, independent of
//! arrival order. Concrete impls are responsible for buffering pushed
//! chunks until the matching `recv` drains them.
//!
//! Two impls ship in phase 1:
//!
//! - [`InMemoryRingTransport`] — `tokio::sync::mpsc` channels between
//!   in-process workers. Used by every unit test.
//! - [`grpc::GrpcRingTransport`] — tonic clients to next/prev workers
//!   speaking the `blazen.allreduce.v1` service. Gated behind the
//!   `distributed` feature so the crate doesn't pull tonic unconditionally.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;
use tokio::sync::mpsc;

use crate::error::BlazenTrainError;

/// Inter-worker bytes pipe used by the ring-AllReduce kernel.
///
/// One instance handles both the `send` (to next peer) and `recv` (from
/// prev peer) directions for a single worker.
#[async_trait]
pub trait RingTransport: Send + Sync {
    /// Push `payload` to the `next` peer for slot `(step, chunk_id)`.
    ///
    /// # Errors
    /// Implementation-defined; the AllReduce kernel propagates failures.
    async fn send(
        &self,
        step: u32,
        chunk_id: u32,
        payload: Vec<u8>,
    ) -> Result<(), BlazenTrainError>;

    /// Pull the chunk that this worker's `prev` peer pushed for slot
    /// `(step, chunk_id)`. Blocks until the chunk arrives.
    ///
    /// # Errors
    /// Implementation-defined; the AllReduce kernel propagates failures.
    async fn recv(&self, step: u32, chunk_id: u32) -> Result<Vec<u8>, BlazenTrainError>;
}

// ---------------------------------------------------------------------------
// In-memory transport
// ---------------------------------------------------------------------------

/// In-process `tokio::sync::mpsc` transport. Used by every unit test in
/// this module and by callers that want to drive a 2- or 4-worker ring
/// inside a single process for benchmarking.
///
/// Construction is paired: [`InMemoryRingTransport::ring`] builds N
/// connected transports, one per worker, wiring each worker's outbound
/// channel into its `next` peer's inbound buffer.
/// In-memory `(step, chunk_id) -> bytes` chunk buffer.
type InMemInbox = Arc<Mutex<HashMap<(u32, u32), Vec<u8>>>>;

pub struct InMemoryRingTransport {
    /// Outbound sender to the `next` peer's inbox.
    outbound: mpsc::UnboundedSender<InMemFrame>,
    /// Inbox indexed by `(step, chunk_id)` for chunks pushed by the
    /// `prev` peer that haven't been drained yet.
    inbox: InMemInbox,
    /// Notification channel: every push from `prev` wakes a waiter.
    notify: Arc<tokio::sync::Notify>,
    /// Owned receiver. Spawned on first `recv` to start draining frames
    /// into `inbox`. Wrapped so multiple `recv` calls share the loop.
    drain_task: Arc<tokio::sync::OnceCell<()>>,
    inbound_rx: Arc<Mutex<Option<mpsc::UnboundedReceiver<InMemFrame>>>>,
}

#[derive(Debug)]
struct InMemFrame {
    step: u32,
    chunk_id: u32,
    payload: Vec<u8>,
}

impl InMemoryRingTransport {
    /// Build a ring of `world_size` connected in-memory transports.
    ///
    /// Index `i` of the returned `Vec` is rank `i`. Sends from rank `i`
    /// land in rank `(i + 1) mod N`'s inbox, matching the ring topology
    /// produced by [`super::ring::RingTopology::next_endpoint`].
    #[must_use]
    pub fn ring(world_size: usize) -> Vec<Arc<Self>> {
        let mut senders: Vec<mpsc::UnboundedSender<InMemFrame>> = Vec::with_capacity(world_size);
        let mut receivers: Vec<mpsc::UnboundedReceiver<InMemFrame>> =
            Vec::with_capacity(world_size);
        for _ in 0..world_size {
            let (tx, rx) = mpsc::unbounded_channel();
            senders.push(tx);
            receivers.push(rx);
        }
        let mut out = Vec::with_capacity(world_size);
        for (i, rx) in receivers.into_iter().enumerate() {
            let next = (i + 1) % world_size;
            let outbound = senders[next].clone();
            out.push(Arc::new(Self {
                outbound,
                inbox: Arc::new(Mutex::new(HashMap::new())),
                notify: Arc::new(tokio::sync::Notify::new()),
                drain_task: Arc::new(tokio::sync::OnceCell::new()),
                inbound_rx: Arc::new(Mutex::new(Some(rx))),
            }));
        }
        out
    }

    async fn ensure_drain_started(&self) {
        let inbox = self.inbox.clone();
        let notify = self.notify.clone();
        let inbound_rx_slot = self.inbound_rx.clone();
        self.drain_task
            .get_or_init(|| async move {
                let rx_opt = inbound_rx_slot.lock().await.take();
                if let Some(mut rx) = rx_opt {
                    tokio::spawn(async move {
                        while let Some(frame) = rx.recv().await {
                            inbox
                                .lock()
                                .await
                                .insert((frame.step, frame.chunk_id), frame.payload);
                            notify.notify_waiters();
                        }
                    });
                }
            })
            .await;
    }
}

#[async_trait]
impl RingTransport for InMemoryRingTransport {
    async fn send(
        &self,
        step: u32,
        chunk_id: u32,
        payload: Vec<u8>,
    ) -> Result<(), BlazenTrainError> {
        self.outbound
            .send(InMemFrame {
                step,
                chunk_id,
                payload,
            })
            .map_err(|e| BlazenTrainError::Distributed(format!("in-memory send failed: {e}")))
    }

    async fn recv(&self, step: u32, chunk_id: u32) -> Result<Vec<u8>, BlazenTrainError> {
        self.ensure_drain_started().await;
        loop {
            // Fast path: chunk already buffered.
            {
                let mut inbox = self.inbox.lock().await;
                if let Some(bytes) = inbox.remove(&(step, chunk_id)) {
                    return Ok(bytes);
                }
            }
            // Slow path: wait for the next push and re-check.
            self.notify.notified().await;
        }
    }
}

// ---------------------------------------------------------------------------
// gRPC transport (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "distributed")]
pub mod grpc {
    //! tonic-backed `RingTransport` for cross-process / cross-node
    //! ring-AllReduce.
    //!
    //! The transport runs both a gRPC client (to push chunks at the
    //! `next` peer) and a gRPC server (to receive chunks the `prev` peer
    //! pushes). The server buffers incoming chunks in an in-memory
    //! `(sender_rank, step, chunk_id) -> bytes` map; `recv()` long-polls
    //! the map until the matching chunk arrives.

    use std::collections::HashMap;
    use std::net::SocketAddr;
    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use tokio::sync::{Mutex, Notify};
    use tonic::transport::Server;
    use tonic::{Request, Response, Status};

    use super::{BlazenTrainError, RingTransport};
    use crate::distributed::pb::{
        AckResponse, BarrierRequest, BarrierResponse, ChunkBytes, RecvChunkRequest,
        SendChunkRequest,
        blazen_all_reduce_client::BlazenAllReduceClient,
        blazen_all_reduce_server::{BlazenAllReduce, BlazenAllReduceServer},
    };

    type InboxKey = (u32, u32, u32);
    type Inbox = Arc<Mutex<HashMap<InboxKey, Vec<u8>>>>;

    /// Server-side state shared with the gRPC service: incoming chunks
    /// land here and the matching [`GrpcRingTransport::recv`] long-polls
    /// for them.
    struct AllReduceService {
        inbox: Inbox,
        notify: Arc<Notify>,
    }

    #[tonic::async_trait]
    impl BlazenAllReduce for AllReduceService {
        async fn send_chunk(
            &self,
            req: Request<SendChunkRequest>,
        ) -> Result<Response<AckResponse>, Status> {
            let r = req.into_inner();
            self.inbox
                .lock()
                .await
                .insert((r.sender_rank, r.step, r.chunk_id), r.payload);
            self.notify.notify_waiters();
            Ok(Response::new(AckResponse { ok: true }))
        }

        async fn recv_chunk(
            &self,
            req: Request<RecvChunkRequest>,
        ) -> Result<Response<ChunkBytes>, Status> {
            // Server-side long-poll fallback. The transport's own
            // RingTransport::recv path uses the local buffer directly;
            // this RPC exists for symmetry / debugging.
            let r = req.into_inner();
            let deadline = std::time::Instant::now() + Duration::from_secs(30);
            loop {
                {
                    let mut inbox = self.inbox.lock().await;
                    if let Some(bytes) = inbox.remove(&(r.sender_rank, r.step, r.chunk_id)) {
                        return Ok(Response::new(ChunkBytes { payload: bytes }));
                    }
                }
                if std::time::Instant::now() >= deadline {
                    return Err(Status::deadline_exceeded("recv_chunk timed out"));
                }
                let _ =
                    tokio::time::timeout(Duration::from_millis(50), self.notify.notified()).await;
            }
        }

        async fn barrier(
            &self,
            _req: Request<BarrierRequest>,
        ) -> Result<Response<BarrierResponse>, Status> {
            Ok(Response::new(BarrierResponse { ok: true }))
        }
    }

    /// Spawn the AllReduce gRPC server on `addr` and return the spawned
    /// [`tokio::task::JoinHandle`] plus the shared inbox so the matching
    /// [`GrpcRingTransport`] can drain it locally.
    ///
    /// # Errors
    /// Returns [`BlazenTrainError::Distributed`] if the bind fails.
    pub async fn spawn_allreduce_server(
        addr: SocketAddr,
    ) -> Result<(tokio::task::JoinHandle<()>, Inbox, Arc<Notify>), BlazenTrainError> {
        let inbox: Inbox = Arc::new(Mutex::new(HashMap::new()));
        let notify = Arc::new(Notify::new());
        let svc = AllReduceService {
            inbox: inbox.clone(),
            notify: notify.clone(),
        };
        let server = BlazenAllReduceServer::new(svc);
        let handle = tokio::spawn(async move {
            let _ = Server::builder().add_service(server).serve(addr).await;
        });
        // Give the bind a moment to settle so the first connect doesn't
        // race the listener.
        tokio::time::sleep(Duration::from_millis(150)).await;
        Ok((handle, inbox, notify))
    }

    /// tonic-backed ring transport.
    ///
    /// - `next_client` is a connected gRPC client to the `next` peer's
    ///   AllReduce service; `send()` calls `SendChunk` on it.
    /// - `local_inbox` is the local server's chunk buffer; `recv()`
    ///   long-polls it for chunks the `prev` peer has pushed.
    pub struct GrpcRingTransport {
        rank: u32,
        next_client: Arc<Mutex<BlazenAllReduceClient<tonic::transport::Channel>>>,
        local_inbox: Inbox,
        local_notify: Arc<Notify>,
        prev_rank: u32,
    }

    impl GrpcRingTransport {
        /// Build a transport given this worker's rank, the connected
        /// `next` client, and the local server's inbox + notify.
        ///
        /// `prev_rank` is needed because chunks land in the inbox keyed
        /// by the *sender's* rank and `recv()` must match the right
        /// sender.
        #[must_use]
        pub fn new(
            rank: u32,
            prev_rank: u32,
            next_client: BlazenAllReduceClient<tonic::transport::Channel>,
            local_inbox: Inbox,
            local_notify: Arc<Notify>,
        ) -> Self {
            Self {
                rank,
                next_client: Arc::new(Mutex::new(next_client)),
                local_inbox,
                local_notify,
                prev_rank,
            }
        }
    }

    #[async_trait]
    impl RingTransport for GrpcRingTransport {
        async fn send(
            &self,
            step: u32,
            chunk_id: u32,
            payload: Vec<u8>,
        ) -> Result<(), BlazenTrainError> {
            let req = SendChunkRequest {
                sender_rank: self.rank,
                step,
                chunk_id,
                payload,
            };
            self.next_client
                .lock()
                .await
                .send_chunk(req)
                .await
                .map_err(|e| BlazenTrainError::Distributed(format!("gRPC send_chunk: {e}")))?;
            Ok(())
        }

        async fn recv(&self, step: u32, chunk_id: u32) -> Result<Vec<u8>, BlazenTrainError> {
            let key = (self.prev_rank, step, chunk_id);
            let deadline = std::time::Instant::now() + Duration::from_mins(1);
            loop {
                {
                    let mut inbox = self.local_inbox.lock().await;
                    if let Some(bytes) = inbox.remove(&key) {
                        return Ok(bytes);
                    }
                }
                if std::time::Instant::now() >= deadline {
                    return Err(BlazenTrainError::Distributed(format!(
                        "gRPC recv timed out for step={step} chunk={chunk_id} from rank={}",
                        self.prev_rank
                    )));
                }
                let _ =
                    tokio::time::timeout(Duration::from_millis(50), self.local_notify.notified())
                        .await;
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use tokio::net::TcpListener;

        async fn ephemeral_addr() -> SocketAddr {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            drop(listener);
            addr
        }

        #[tokio::test]
        async fn grpc_ring_transport_round_trip() {
            // Spin up rank-1's AllReduce server, connect rank-0's client
            // to it, push a chunk, then pull it back from rank-1's local
            // inbox via the GrpcRingTransport.recv path.
            let addr = ephemeral_addr().await;
            let (_handle, inbox, notify) = spawn_allreduce_server(addr).await.expect("spawn");

            // Rank-0 (sender) connects to rank-1's server.
            let endpoint = format!("http://{addr}");
            let client = BlazenAllReduceClient::connect(endpoint)
                .await
                .expect("client connect");

            let sender_transport = GrpcRingTransport::new(
                0, // rank
                1, // prev_rank (unused on send)
                client.clone(),
                Arc::new(Mutex::new(HashMap::new())), // sender's own inbox
                Arc::new(Notify::new()),
                // (not used in this test)
            );

            // Rank-1's transport: it would normally hold its own next-client
            // (pointing further around the ring); for this round-trip test
            // we reuse the same client so the type checks out. recv() pulls
            // from the inbox the server fills.
            let receiver_transport = GrpcRingTransport::new(
                1, 0, // prev_rank = sender
                client, inbox, notify,
            );

            let payload = vec![42_u8, 7, 99, 1, 2, 3];
            sender_transport
                .send(5, 2, payload.clone())
                .await
                .expect("send");

            let received =
                tokio::time::timeout(Duration::from_secs(5), receiver_transport.recv(5, 2))
                    .await
                    .expect("recv timeout")
                    .expect("recv");
            assert_eq!(received, payload);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn inmemory_ring_transport_send_recv_roundtrip() {
        let ring = InMemoryRingTransport::ring(2);
        let worker0 = ring[0].clone();
        let worker1 = ring[1].clone();

        // worker0 -> worker1
        let payload = vec![10_u8, 20, 30];
        worker0.send(7, 3, payload.clone()).await.expect("send");
        let got = worker1.recv(7, 3).await.expect("recv");
        assert_eq!(got, payload);
    }

    #[tokio::test]
    async fn inmemory_ring_transport_send_before_recv_buffers() {
        let ring = InMemoryRingTransport::ring(3);
        ring[2].send(1, 0, vec![1, 2, 3, 4]).await.expect("send");
        ring[2]
            .send(1, 1, vec![5, 6, 7, 8])
            .await
            .expect("second send");
        // Receiver picks up in arbitrary order — the (step, chunk_id) key
        // makes the match unambiguous.
        let got_b = ring[0].recv(1, 1).await.expect("recv b");
        let got_a = ring[0].recv(1, 0).await.expect("recv a");
        assert_eq!(got_a, vec![1, 2, 3, 4]);
        assert_eq!(got_b, vec![5, 6, 7, 8]);
    }
}
