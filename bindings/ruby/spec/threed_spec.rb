# frozen_string_literal: true

require "spec_helper"
require "json"
require "socket"

# Ruby 3.4+ removed the +base64+ stdlib from default gems. Use the
# Array#pack codec directly — it's identical to
# ThreeDSpecBase64.encode for binary inputs and avoids forcing every
# downstream gem author to add `base64` to their Gemfile.
module ThreeDSpecBase64
  module_function

  def encode(bytes)
    [bytes].pack("m0").force_encoding(Encoding::US_ASCII)
  end
end

# The +async+ gem is a dev dep; it may be absent on slim CI images.
# Mirrors the +ASYNC_AVAILABLE+ probe in +spec/blazen_spec.rb+ so this
# file is self-contained.
begin
  require "async" unless defined?(Async)
  THREED_ASYNC_AVAILABLE = true
rescue LoadError
  THREED_ASYNC_AVAILABLE = false
end

# Spec for the +Blazen::ThreeD+ Compat3dProvider Ruby wrapper.
#
# These specs spin up a minimal in-process HTTP server (raw TCPServer,
# no WEBrick / no Rack — those gems aren't dev-deps) and point the
# provider at it. The server hands back canned JSON payloads in the
# same shape the +blazen-3d+ Compat3dProvider expects from a real
# upstream service:
#
#   - +texturize+ → +{ textured_glb_b64, mime_type, pbr_maps? }+
#   - +rig+        → +{ rigged_glb_b64, mime_type, bone_names }+
#   - +refine+     → +{ refined_glb_b64, mime_type, stats }+
#   - +animate+    → +{ animated_glb_b64, mime_type, duration_seconds, fps }+
#
# Real network calls are NOT made. The Fiber.scheduler spec uses the
# +async+ gem to verify cooperative scheduling.

# Stub HTTP server backed by raw TCPServer. Responds to every request
# with the +responder+ block's return value (a +[status, body_json]+
# pair). Recorded requests are appended to +@requests+ for assertion.
class StubThreeDServer
  attr_reader :requests, :port

  def initialize(&responder)
    @responder = responder
    @requests = []
    @server = TCPServer.new("127.0.0.1", 0)
    @port = @server.addr[1]
    @thread = Thread.new { accept_loop }
  end

  def base_url
    "http://127.0.0.1:#{@port}"
  end

  def stop
    @server.close
  rescue StandardError
    nil
  ensure
    @thread.kill if @thread&.alive?
  end

  private

  def accept_loop
    loop do
      client = @server.accept
      handle_client(client)
    rescue IOError, Errno::EBADF
      break
    end
  rescue StandardError
    nil
  end

  def handle_client(client)
    # Read the request line + headers
    request_line = client.gets
    return if request_line.nil?

    headers = {}
    while (line = client.gets) && line.strip != ""
      key, value = line.split(":", 2)
      headers[key.strip.downcase] = value.strip if key && value
    end

    length = (headers["content-length"] || "0").to_i
    body = length.positive? ? client.read(length) : ""

    method, path, = request_line.split(" ")
    @requests << {
      method: method, path: path, headers: headers, body: body,
    }

    status, json_body = @responder.call(method, path, headers, body)
    write_response(client, status, json_body)
  ensure
    client.close rescue nil
  end

  def write_response(client, status, body)
    body = body.to_s
    client.write([
      "HTTP/1.1 #{status} OK",
      "Content-Type: application/json",
      "Content-Length: #{body.bytesize}",
      "Connection: close",
      "",
      body,
    ].join("\r\n"))
  end
end

RSpec.describe Blazen::ThreeD, if: Blazen::ThreeD.available? do
  # Tiny synthetic "GLB" payload — the provider doesn't inspect it.
  let(:dummy_glb) { "glTF\x02\x00\x00\x00stub-mesh-bytes".b }
  let(:output_glb) { "glTF\x02\x00\x00\x00stub-out-bytes".b }

  before(:all) { Blazen.init }

  describe ".available?" do
    it "is true when the native lib carries the threed-compat-proxy feature" do
      expect(Blazen::ThreeD.available?).to be(true)
    end
  end

  describe Blazen::ThreeD::Compat3dProvider do
    it "constructs and frees a provider against a stub URL" do
      provider = described_class.new(
        base_url: "http://127.0.0.1:1",
        api_key: "sk-test",
        timeout_secs: 5,
      )
      expect(provider.ptr).not_to be_null
    end

    it "rejects nil / empty base_url" do
      expect { described_class.new(base_url: nil) }.to raise_error(Blazen::ValidationError)
      expect { described_class.new(base_url: "") }.to raise_error(Blazen::ValidationError)
    end

    context "with a stub HTTP backend" do
      let(:pbr_albedo_png) { ("\x89PNG\r\n" + "albedo-stub").b }

      let(:server) do
        StubThreeDServer.new do |_method, path, headers, _body|
          case path
          when "/v1/3d/texturize"
            # Echo back an Authorization-header presence flag so we can assert on it.
            body = {
              textured_glb_b64: ThreeDSpecBase64.encode(output_glb),
              mime_type: "model/gltf-binary",
              pbr_maps: {
                albedo_png_b64: ThreeDSpecBase64.encode(pbr_albedo_png),
                normal_png_b64: nil,
              },
              _saw_auth: headers["authorization"],
            }
            [200, JSON.generate(body)]
          when "/v1/3d/rig"
            body = {
              rigged_glb_b64: ThreeDSpecBase64.encode(output_glb),
              mime_type: "model/gltf-binary",
              bone_names: %w[root spine head],
            }
            [200, JSON.generate(body)]
          when "/v1/3d/refine"
            body = {
              refined_glb_b64: ThreeDSpecBase64.encode(output_glb),
              mime_type: "model/gltf-binary",
              stats: {
                input_tri_count: 1000,
                output_tri_count: 500,
                uv_chart_count: 12,
              },
            }
            [200, JSON.generate(body)]
          when "/v1/3d/animate"
            body = {
              animated_glb_b64: ThreeDSpecBase64.encode(output_glb),
              mime_type: "model/gltf-binary",
              duration_seconds: 4.5,
              fps: 30,
            }
            [200, JSON.generate(body)]
          else
            [404, "{}"]
          end
        end
      end

      after { server.stop }

      let(:provider) do
        described_class.new(
          base_url: server.base_url,
          api_key: "sk-stub-token",
          timeout_secs: 10,
        )
      end

      it "round-trips texturize: PBR bytes survive the base64 wire" do
        result = provider.texturize(
          dummy_glb,
          prompt: "weathered bronze",
          reference_image: "\x89PNG\r\nref",
          pbr: true,
          resolution: 1024,
        )
        expect(result).to be_a(Blazen::ThreeD::Result)
        expect(result.stage).to eq(:texturize)
        expect(result.mime_type).to eq("model/gltf-binary")
        expect(result.glb_bytes).to eq(output_glb)
        expect(result.pbr_maps?).to be(true)
        expect(result.pbr_map(:albedo)).to eq(pbr_albedo_png)
        expect(result.pbr_map(:normal)).to be_nil
      end

      it "attaches the bearer token on the outbound request" do
        provider.texturize(dummy_glb)
        # The stub appends the header presence into the response, but the
        # cleanest assertion is on the recorded request headers.
        auth = server.requests.last[:headers]["authorization"]
        expect(auth).to eq("Bearer sk-stub-token")
      end

      it "round-trips rig: bone_names list parsed correctly" do
        result = provider.rig(dummy_glb, template: "humanoid", skin: true)
        expect(result.stage).to eq(:rig)
        expect(result.bone_names).to eq(%w[root spine head])
        expect(result.pbr_maps?).to be(false)
        expect(result.refine_stats).to be_nil
      end

      it "round-trips refine: stats decoded from JSON" do
        result = provider.refine(
          dummy_glb,
          decimate_target_tris: 500,
          unwrap_uvs: true,
        )
        expect(result.stage).to eq(:refine)
        stats = result.refine_stats
        expect(stats[:input_tri_count]).to eq(1000)
        expect(stats[:output_tri_count]).to eq(500)
        expect(stats[:uv_chart_count]).to eq(12)
      end

      it "round-trips animate: duration + fps surfaced" do
        result = provider.animate(
          dummy_glb,
          prompt: "walks forward",
          duration_seconds: 5.0,
          fps: 24,
          loop_animation: true,
        )
        expect(result.stage).to eq(:animate)
        expect(result.duration_seconds).to be_within(0.01).of(4.5)
        expect(result.fps).to eq(30)
      end

      it "raises ValidationError for an unknown PBR channel" do
        result = provider.texturize(dummy_glb, pbr: true)
        expect { result.pbr_map(:bogus) }.to raise_error(Blazen::ValidationError)
      end

      it "surfaces an HTTP-500 error as a Blazen::Error" do
        error_server = StubThreeDServer.new do |_method, _path, _h, _b|
          [500, JSON.generate(error: "synthetic backend failure")]
        end
        begin
          err_provider = described_class.new(
            base_url: error_server.base_url, timeout_secs: 5,
          )
          expect { err_provider.texturize(dummy_glb) }
            .to raise_error(Blazen::Error)
        ensure
          error_server.stop
        end
      end

      it "composes with Fiber.scheduler under Async {}", :aggregate_failures do
        unless THREED_ASYNC_AVAILABLE
          skip "install the +async+ gem to exercise Fiber.scheduler interop (gem install async)"
        end

        # Three concurrent texturize calls should overlap under the
        # async reactor — the in-process server handles each on its own
        # accept thread, so a Fiber.scheduler that yields on the future
        # fd lets the runs proceed concurrently.
        results = Async do
          tasks = Array.new(3) do |i|
            Async { provider.texturize(dummy_glb, prompt: "p-#{i}") }
          end
          tasks.map(&:wait)
        end.wait

        expect(results.length).to eq(3)
        results.each do |r|
          expect(r).to be_a(Blazen::ThreeD::Result)
          expect(r.glb_bytes).to eq(output_glb)
        end
      end
    end
  end
end
