import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    kotlin("jvm") version "2.1.20"
    kotlin("plugin.serialization") version "2.1.20"
    `java-library`
}

group = "dev.zorpx.blazen"
version = "0.1.0"

repositories {
    mavenCentral()
}

kotlin {
    // Warning (not error) so the UniFFI-generated `blazen.kt`, which does not
    // follow explicit-API conventions, doesn't fail the build. Our hand-
    // written wrapper sources do mark every exported declaration `public`.
    explicitApiWarning()

    compilerOptions {
        // JVM 17 is our minimum target. We don't pin a toolchain JDK here so
        // the build runs against whatever JDK 17+ is on PATH (the project
        // CLAUDE.md uses JDK 26; CI can run JDK 17).
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

dependencies {
    api("net.java.dev.jna:jna:5.14.0")
    api("org.jetbrains.kotlinx:kotlinx-coroutines-core:1.8.0")
    api("org.jetbrains.kotlinx:kotlinx-serialization-json:1.6.0")

    testImplementation("org.junit.jupiter:junit-jupiter:5.10.0")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.8.0")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.test {
    useJUnitPlatform()
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
    }
}

