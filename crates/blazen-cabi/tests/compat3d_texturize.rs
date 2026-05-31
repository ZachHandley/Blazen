//! Regression test for the unified `Compat3dProvider` texturize path.
//!
//! Guards against the prior bug where `Compat3dProvider::texturize` returned
//! `Unsupported` instead of delegating to the HTTP post-processing backend.
//! `blazen-cabi` is a `cdylib`/`staticlib` (no rlib), so this drives the
//! `blazen_uniffi` layer that the cabi `blazen_compat3d_*` symbols wrap,
//! against a one-shot mock HTTP server.
#![cfg(feature = "threed-compat-proxy")]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::time::{Duration, Instant};

use blazen_uniffi::concrete::three_d::{Compat3dProvider, TexturizeRequest};

#[test]
fn texturize_hits_backend_not_unsupported() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    // Bounded one-shot mock server: a non-blocking accept loop with a hard
    // deadline so the thread ALWAYS terminates (and `handle.join()` below can
    // never hang) even if the client never connects — e.g. a future
    // regression where `texturize` stops delegating to the HTTP backend.
    let handle = std::thread::spawn(move || {
        listener.set_nonblocking(true).ok();
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            match listener.accept() {
                Ok((mut s, _)) => {
                    s.set_read_timeout(Some(Duration::from_secs(5))).ok();
                    let mut buf = [0u8; 8192];
                    let _ = s.read(&mut buf);
                    // base64 "AAEC" decodes to bytes [0, 1, 2].
                    let body = r#"{"textured_glb_b64":"AAEC","mime_type":"model/gltf-binary"}"#;
                    let resp = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    );
                    let _ = s.write_all(resp.as_bytes());
                    let _ = s.flush();
                    return;
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        return;
                    }
                    std::thread::sleep(Duration::from_millis(20));
                }
                Err(_) => return,
            }
        }
    });

    let provider = Compat3dProvider::new(format!("http://127.0.0.1:{port}"), None, Some(10));
    let req = TexturizeRequest {
        prompt: None,
        reference_image: None,
        style: None,
        resolution: None,
        pbr: false,
    };
    let res = provider.texturize_blocking(vec![1, 2, 3], req);
    let _ = handle.join();
    match res {
        Ok(r) => assert_eq!(r.textured_glb, vec![0, 1, 2]),
        Err(e) => panic!("texturize returned error: {e:?}"),
    }
}
