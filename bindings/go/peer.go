package blazen

import (
	"context"
	"encoding/json"
	"fmt"
	"runtime"
	"sync"
	"time"

	uniffiblazen "github.com/zachhandley/Blazen/bindings/go/internal/uniffi/blazen"
)

// PeerClient is a client handle for invoking workflows on a remote
// [PeerServer] over gRPC. RPCs go out over a multiplexed HTTP/2 channel
// held inside the client; multiple concurrent calls on the same
// PeerClient are safe and share the connection.
//
// Construct one via [ConnectPeerClient]. Call [PeerClient.Close] when
// finished to release the underlying native handle; a finalizer is
// attached as a safety net, but explicit Close is preferred for
// predictable resource release.
type PeerClient struct {
	inner *uniffiblazen.PeerClient
	once  sync.Once
}

// ConnectPeerClient establishes a connection to a remote Blazen peer.
//
// address is the peer's gRPC endpoint URI (for example
// "http://10.0.0.1:50051" or "https://node-a.local:7443").
// clientNodeID is a local identifier stamped into outgoing requests for
// trace correlation on both sides of the connection; the local hostname
// or a process-startup UUID work well.
//
// This call blocks while the TCP/HTTP-2 handshake completes — the
// upstream FFI constructor drives the connect on the shared Tokio
// runtime so synchronous Go code can set up a client without an async
// story. A [PeerError] with Kind "Transport" is returned if the URI is
// invalid or the handshake fails.
func ConnectPeerClient(address, clientNodeID string) (*PeerClient, error) {
	ensureInit()
	inner, err := uniffiblazen.PeerClientConnect(address, clientNodeID)
	if err != nil {
		return nil, wrapErr(err)
	}
	c := &PeerClient{inner: inner}
	// Safety net: if the caller forgets Close, the finalizer reclaims the
	// underlying native handle on GC. Close() clears the finalizer via
	// the underlying Destroy().
	runtime.SetFinalizer(c, func(pc *PeerClient) { pc.Close() })
	return c, nil
}

// NodeID returns the node-id this client stamps into outgoing requests
// for trace correlation. Returns the empty string after [PeerClient.Close].
func (c *PeerClient) NodeID() string {
	if c == nil || c.inner == nil {
		return ""
	}
	return c.inner.NodeId()
}

// RunRemoteWorkflow invokes a workflow on the remote peer and waits for
// its terminal result.
//
// workflowName is the symbolic name the remote peer's step registry
// knows the workflow as. stepIDs lists the step identifiers to execute
// in order; every entry must be registered on the remote peer's process
// or the call fails with a [PeerError] of Kind "UnknownStep". input is
// JSON-marshalled into the workflow's entry payload — passing nil sends
// the JSON literal `null`. timeout bounds the remote workflow's
// wall-clock execution; pass 0 to defer to the peer's default deadline.
// Sub-second durations are rounded up to the next whole second.
//
// The Rust side of the call is blocking from Go's perspective. To
// honour ctx cancellation the FFI call runs on a background goroutine
// and the function selects on ctx.Done(); when ctx fires first the
// function returns ctx.Err() immediately. The remote workflow itself
// keeps running until it finishes naturally — cancellation propagation
// across the wire is a known gap pending an upstream UniFFI feature.
//
// The returned [WorkflowResult] carries the terminal payload
// synthesised from the remote response. Per-run LLM token usage and
// cost are not propagated over the wire and the returned result reports
// them as zero; callers needing those should query the remote peer's
// telemetry directly.
func (c *PeerClient) RunRemoteWorkflow(ctx context.Context, workflowName string, stepIDs []string, input any, timeout time.Duration) (*WorkflowResult, error) {
	payload, err := json.Marshal(input)
	if err != nil {
		return nil, &ValidationError{Message: fmt.Sprintf("marshal remote workflow input: %s", err)}
	}
	return c.RunRemoteWorkflowJSON(ctx, workflowName, stepIDs, string(payload), timeout)
}

// RunRemoteWorkflowJSON is the lower-level form of
// [PeerClient.RunRemoteWorkflow] taking a pre-JSON payload string. Use
// this when the input is already a JSON document and the extra
// marshal/unmarshal of RunRemoteWorkflow would be wasteful.
//
// ctx cancellation, timeout, and error semantics match
// [PeerClient.RunRemoteWorkflow].
func (c *PeerClient) RunRemoteWorkflowJSON(ctx context.Context, workflowName string, stepIDs []string, inputJSON string, timeout time.Duration) (*WorkflowResult, error) {
	if c == nil || c.inner == nil {
		return nil, &ValidationError{Message: "peer client has been closed"}
	}
	timeoutSecs := durationToSecondsPtr(timeout)

	type runResult struct {
		res uniffiblazen.WorkflowResult
		err error
	}
	done := make(chan runResult, 1)
	go func() {
		res, err := c.inner.RunRemoteWorkflow(workflowName, stepIDs, inputJSON, timeoutSecs)
		done <- runResult{res: res, err: err}
	}()

	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case r := <-done:
		if r.err != nil {
			return nil, wrapErr(r.err)
		}
		out := workflowResultFromFFI(r.res)
		return &out, nil
	}
}

// Close releases the underlying native handle, closing the gRPC
// connection. It is safe to call Close multiple times and from multiple
// goroutines; subsequent calls are no-ops.
func (c *PeerClient) Close() {
	c.once.Do(func() {
		if c.inner != nil {
			c.inner.Destroy()
			c.inner = nil
		}
	})
}

// PeerServer is a node-local Blazen peer gRPC server that accepts
// inbound workflow requests from remote peers.
//
// Construct one with [NewPeerServer] and start the listener with
// [PeerServer.Serve] or [PeerServer.ServeBlocking]. Dispatched
// workflows are resolved at request time through the process-wide step
// registry, so any workflow whose steps have been registered in this
// process can be invoked remotely by name.
//
// Call [PeerServer.Close] when finished to release the underlying
// native handle; a finalizer is attached as a safety net, but explicit
// Close is preferred for predictable resource release.
type PeerServer struct {
	inner *uniffiblazen.PeerServer
	once  sync.Once
}

// NewPeerServer constructs an inbound peer server identifiable as
// nodeID. The nodeID is the stable identifier this server stamps onto
// every remote-ref descriptor it returns; the hostname or a UUID picked
// at process startup work well.
//
// The server is not listening yet — call [PeerServer.Serve] or
// [PeerServer.ServeBlocking] to bind. The constructor itself is
// infallible.
func NewPeerServer(nodeID string) *PeerServer {
	ensureInit()
	s := &PeerServer{inner: uniffiblazen.NewPeerServer(nodeID)}
	// Safety net: if the caller forgets Close, the finalizer reclaims the
	// underlying native handle on GC. Close() clears the finalizer via
	// the underlying Destroy().
	runtime.SetFinalizer(s, func(ps *PeerServer) { ps.Close() })
	return s
}

// Serve binds the gRPC server to listenAddress (for example ":50051" or
// "0.0.0.0:50051") and blocks serving requests until the underlying
// socket fails or the server is consumed.
//
// listenAddress must parse as a standard socket address. The underlying
// server is consumed by this call; invoking Serve or ServeBlocking a
// second time on the same PeerServer returns a [ValidationError]. A
// [PeerError] with Kind "Transport" is returned if the listener fails
// to bind or hits a fatal I/O error while serving.
//
// The Rust side of the call is blocking from Go's perspective. To
// honour ctx cancellation the FFI call runs on a background goroutine
// and the function selects on ctx.Done(); on cancellation the wait
// unblocks immediately, but the Rust-side server keeps running until it
// finishes naturally (cancellation propagation into the runtime is a
// known gap pending an upstream UniFFI feature).
func (s *PeerServer) Serve(ctx context.Context, listenAddress string) error {
	if s == nil || s.inner == nil {
		return &ValidationError{Message: "peer server has been closed"}
	}
	done := make(chan error, 1)
	go func() {
		done <- s.inner.Serve(listenAddress)
	}()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case err := <-done:
		if err != nil {
			return wrapErr(err)
		}
		return nil
	}
}

// ServeBlocking is the synchronous form of [PeerServer.Serve]. It
// blocks the calling goroutine until the server exits.
//
// Error semantics match [PeerServer.Serve].
func (s *PeerServer) ServeBlocking(listenAddress string) error {
	if s == nil || s.inner == nil {
		return &ValidationError{Message: "peer server has been closed"}
	}
	if err := s.inner.ServeBlocking(listenAddress); err != nil {
		return wrapErr(err)
	}
	return nil
}

// Close releases the underlying native handle. It is safe to call Close
// multiple times and from multiple goroutines; subsequent calls are
// no-ops.
func (s *PeerServer) Close() {
	s.once.Do(func() {
		if s.inner != nil {
			s.inner.Destroy()
			s.inner = nil
		}
	})
}

// durationToSecondsPtr converts a Go time.Duration into the optional
// uint64 second count expected by the FFI's timeout parameter. A
// non-positive duration maps to nil (deferring to the remote peer's
// default deadline); sub-second positive durations round up so a
// caller-supplied 500ms timeout still allows the remote a full second.
func durationToSecondsPtr(d time.Duration) *uint64 {
	if d <= 0 {
		return nil
	}
	secs := uint64((d + time.Second - 1) / time.Second)
	return &secs
}
