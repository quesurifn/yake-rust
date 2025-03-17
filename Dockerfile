# Stage 1: Common builder base
FROM rust:latest as builder-base
WORKDIR /usr/src/app

# Update certificates
RUN update-ca-certificates

# First copy dependency manifests and build the dependency tree
COPY ./yake_rust/Cargo.toml ./yake_rust/Cargo.toml
COPY ./yake_rust/README.md ./yake_rust/README.md
COPY ./server/Cargo.toml ./server/Cargo.toml
COPY ./Cargo.toml ./Cargo.toml
COPY ./Cargo.lock ./Cargo.lock

# Create empty source files to trick cargo into building our dependencies
RUN mkdir -p ./yake_rust/src && \
    touch ./yake_rust/src/lib.rs && \
    mkdir -p ./yake_rust/benches && \
    touch ./yake_rust/benches/bench.rs && \
    mkdir -p ./server/src && \
    echo 'fn main() {}' > ./server/src/main.rs && \
    mkdir -p ./python && \
    touch ./python/Cargo.toml && \
    mkdir -p ./python/src && \
    touch ./python/src/lib.rs

# Copy the actual source code
COPY ./yake_rust/src ./yake_rust/src
COPY ./yake_rust/benches ./yake_rust/benches
COPY ./server/src ./server/src
COPY ./python/Cargo.toml ./python/
COPY ./python/src ./python/src

# Stage 2: x86_64 builder
FROM builder-base as builder-x86_64
RUN apt-get update && \
    apt-get install -y musl-tools musl-dev && \
    rustup target add x86_64-unknown-linux-musl

# Build for x86_64
RUN cargo build --release --target x86_64-unknown-linux-musl --package server

# Stage 3: aarch64 builder
FROM builder-base as builder-aarch64
RUN apt-get update && \
    apt-get install -y musl-tools musl-dev gcc-aarch64-linux-gnu && \
    rustup target add aarch64-unknown-linux-musl

# Set environment variables for cross-compilation
ENV CC_aarch64_unknown_linux_musl=aarch64-linux-gnu-gcc
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_MUSL_LINKER=aarch64-linux-gnu-gcc

# Build for aarch64
RUN cargo build --release --target aarch64-unknown-linux-musl --package server

# Stage 4: Final runtime image for x86_64
FROM alpine:3.19 AS runtime-x86_64
# Install necessary runtime dependencies
RUN apk --no-cache add ca-certificates
# Create a non-root user to run the application
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
WORKDIR /app
# Define build argument for port with a default value
ARG PORT=8080
ENV SRV_PORT=${PORT}
ENV RUST_LOG=info
ENV WORKERS=4
# Copy the binary from the x86_64 builder stage
COPY --from=builder-x86_64 /usr/src/app/target/x86_64-unknown-linux-musl/release/server /app/server
RUN chown appuser:appgroup /app/server
RUN chmod +x /app/server
# Switch to the non-root user
USER appuser
# Expose the port specified by the build argument
EXPOSE ${PORT}
# Command to run the application
CMD ["/app/server"]

# Stage 5: Final runtime image for aarch64
FROM alpine:3.19 AS runtime-aarch64
# Install necessary runtime dependencies
RUN apk --no-cache add ca-certificates
# Create a non-root user to run the application
RUN addgroup -S appgroup && adduser -S appuser -G appgroup
WORKDIR /app
# Define build argument for port with a default value
ARG PORT=8080
ENV SRV_PORT=${PORT}
ENV RUST_LOG=info
ENV WORKERS=4
# Copy the binary from the aarch64 builder stage
COPY --from=builder-aarch64 /usr/src/app/target/aarch64-unknown-linux-musl/release/server /app/server
RUN chown appuser:appgroup /app/server
RUN chmod +x /app/server
# Switch to the non-root user
USER appuser
# Expose the port specified by the build argument
EXPOSE ${PORT}
# Command to run the application
CMD ["/app/server"]

# Use a build argument to select the appropriate final image
FROM runtime-${TARGETARCH} AS runtime