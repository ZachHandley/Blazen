# frozen_string_literal: true

require "spec_helper"

# Smoke tests for the control-plane bindings.
#
# These specs deliberately use only the surface that does NOT require a
# live control-plane server. The full integration test lives in the
# Rust workspace ({@code crates/blazen-controlplane/tests}); here we
# verify that the FFI surface is wired up, the Ruby wrappers raise the
# right exception classes on the failure paths, and the module's
# availability check works.
RSpec.describe Blazen::ControlPlane do
  before(:all) { Blazen.init }

  describe ".available?" do
    it "returns true when the native library was built with the distributed feature" do
      expect(Blazen::ControlPlane.available?).to be(true)
    end
  end

  describe ".status_symbol" do
    it "maps each BLAZEN_RUN_STATUS_* constant to its canonical symbol" do
      expect(Blazen::ControlPlane.status_symbol(0)).to eq(:pending)
      expect(Blazen::ControlPlane.status_symbol(1)).to eq(:running)
      expect(Blazen::ControlPlane.status_symbol(2)).to eq(:completed)
      expect(Blazen::ControlPlane.status_symbol(3)).to eq(:failed)
      expect(Blazen::ControlPlane.status_symbol(4)).to eq(:cancelled)
      expect(Blazen::ControlPlane.status_symbol(99)).to eq(:unknown)
    end
  end

  describe Blazen::ControlPlane::Client do
    it "rejects a connect to an unparseable endpoint" do
      expect {
        Blazen::ControlPlane::Client.connect_blocking("not a uri")
      }.to raise_error(Blazen::Error)
    end

    it "rejects an async connect to an unparseable endpoint" do
      expect {
        Blazen::ControlPlane::Client.connect("not a uri")
      }.to raise_error(Blazen::Error)
    end

    it "rejects a connect to an unreachable endpoint" do
      # An RFC-5737 documentation address ensures the connect fails at
      # the TCP layer rather than hitting a real service. tonic surfaces
      # the failure as a Transport error which the cabi wraps into a
      # PeerError (kind=ControlPlaneTransport).
      expect {
        Blazen::ControlPlane::Client.connect_blocking("http://192.0.2.1:1")
      }.to raise_error(Blazen::Error)
    end
  end

  describe Blazen::ControlPlane::Worker do
    it "rejects construction against a malformed endpoint" do
      expect {
        Blazen::ControlPlane::Worker.new(
          endpoint: "not a uri",
          node_id:  "test-node",
        ) { |_assignment| {} }
      }.to raise_error(Blazen::Error)
    end

    it "requires a handler or a block" do
      expect {
        Blazen::ControlPlane::Worker.new(
          endpoint: "http://127.0.0.1:7445",
          node_id:  "test-node",
        )
      }.to raise_error(ArgumentError, /handler.*or a block/)
    end

    it "rejects passing both a handler and a block" do
      handler = Object.new
      expect {
        Blazen::ControlPlane::Worker.new(
          endpoint: "http://127.0.0.1:7445",
          node_id:  "test-node",
          handler:  handler,
        ) { |_| {} }
      }.to raise_error(ArgumentError, /handler.*OR a block/)
    end

    it "rejects an unknown admission mode" do
      expect {
        Blazen::ControlPlane::Worker.new(
          endpoint:  "http://127.0.0.1:7445",
          node_id:   "test-node",
          admission: :bogus,
        ) { |_| {} }
      }.to raise_error(ArgumentError, /unknown admission mode/)
    end

    describe "mTLS" do
      it "rejects an incomplete mtls: hash" do
        expect {
          Blazen::ControlPlane::Worker.new(
            endpoint: "https://127.0.0.1:7445",
            node_id:  "test-node",
            mtls:     { cert_path: "/tmp/cert.pem", key_path: "/tmp/key.pem" },
          ) { |_| {} }
        }.to raise_error(ArgumentError, /ca_path/)
      end

      it "rejects when the PEM files don't exist" do
        expect {
          Blazen::ControlPlane::Worker.new_with_mtls(
            endpoint:  "https://127.0.0.1:7445",
            node_id:   "test-node",
            cert_path: "/nonexistent/cert.pem",
            key_path:  "/nonexistent/key.pem",
            ca_path:   "/nonexistent/ca.pem",
          ) { |_| {} }
        }.to raise_error(Blazen::Error)
      end
    end
  end

  describe "subscriptions" do
    it "rejects subscribe_run_events without a block" do
      # We need a Client to call .subscribe_run_events on, but
      # constructing one requires a server. The wrapper's nil-block
      # guard runs before the FFI call, so use a stub Client whose
      # internal @ptr is the NULL pointer — the ArgumentError fires
      # before we touch the cabi.
      client = Blazen::ControlPlane::Client.allocate
      client.instance_variable_set(:@ptr, ::FFI::Pointer::NULL)
      expect {
        client.subscribe_run_events("00000000-0000-0000-0000-000000000000")
      }.to raise_error(ArgumentError, /block/)
    end

    it "rejects subscribe_all without a block" do
      client = Blazen::ControlPlane::Client.allocate
      client.instance_variable_set(:@ptr, ::FFI::Pointer::NULL)
      expect {
        client.subscribe_all(required_tags: [])
      }.to raise_error(ArgumentError, /block/)
    end
  end

  describe "mTLS connect" do
    it "rejects when the PEM files don't exist" do
      expect {
        Blazen::ControlPlane::Client.connect_with_mtls_blocking(
          "https://127.0.0.1:1",
          cert_path: "/nonexistent/cert.pem",
          key_path:  "/nonexistent/key.pem",
          ca_path:   "/nonexistent/ca.pem",
        )
      }.to raise_error(Blazen::Error)
    end

    it "rejects an async connect when PEM files don't exist" do
      expect {
        Blazen::ControlPlane::Client.connect_with_mtls(
          "https://127.0.0.1:1",
          cert_path: "/nonexistent/cert.pem",
          key_path:  "/nonexistent/key.pem",
          ca_path:   "/nonexistent/ca.pem",
        )
      }.to raise_error(Blazen::Error)
    end
  end
end
