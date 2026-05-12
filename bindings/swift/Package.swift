// swift-tools-version:5.9
//
// SwiftPM manifest for the Blazen Swift binding.
//
// This package exposes Blazen as a Swift module with three targets layered
// from low to high level:
//
// - `BlazenFFI` — a C target that publishes the UniFFI-generated header and
//   module map so Swift code can import the raw FFI entry points.
// - `UniFFIBlazen` — the UniFFI-generated Swift glue compiled as a normal
//   Swift target. It imports `BlazenFFI`, re-exports the lifted types, and
//   provides the building blocks (records, errors, opaque handles) that the
//   idiomatic wrapper sits on top of.
// - `BlazenSwift` — the public, idiomatic Swift API. Callers depend on this
//   target and never see UniFFI-specific names directly.
//
// Native linkage strategy:
//
// On Linux / local-dev the host already builds `libblazen_uniffi.{so,a}` in
// `target/release/` next to the rest of the workspace. Until the cross-
// platform XCFramework lands in Phase H, we link against that host copy via
// `linkerSettings: [.linkedLibrary, .unsafeFlags(["-L", ...])]`. Apple
// platform builds will swap that for a binary target pointing at the
// per-arch XCFramework.

import PackageDescription

let package = Package(
    name: "BlazenSwift",
    platforms: [
        .macOS(.v12),
        .iOS(.v15),
        .tvOS(.v15),
        .watchOS(.v8),
    ],
    products: [
        .library(
            name: "BlazenSwift",
            targets: ["BlazenSwift"]
        ),
        .library(
            name: "UniFFIBlazen",
            targets: ["UniFFIBlazen"]
        ),
    ],
    targets: [
        .systemLibrary(
            name: "BlazenFFI",
            path: "Sources/BlazenFFI"
        ),
        .target(
            name: "UniFFIBlazen",
            dependencies: ["BlazenFFI"],
            path: "Sources/UniFFIBlazen",
            linkerSettings: [
                .linkedLibrary("blazen_uniffi"),
                .unsafeFlags(["-L../../target/release"], .when(platforms: [.linux])),
                .unsafeFlags(["-L../../target/release"], .when(platforms: [.macOS])),
            ]
        ),
        .target(
            name: "BlazenSwift",
            dependencies: ["UniFFIBlazen"],
            path: "Sources/BlazenSwift"
        ),
        .testTarget(
            name: "BlazenSwiftTests",
            dependencies: ["BlazenSwift"],
            path: "Tests/BlazenSwiftTests"
        ),
    ]
)
