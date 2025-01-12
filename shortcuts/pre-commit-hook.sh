#!/bin/bash

set -ex

# Run tests.
export RUST_BACKTRACE=1
cargo test --all-features

# Is the formatting correct?
cargo +nightly fmt -- --check

# Doest the code compile? The warnings should be fixed before committing.
cargo check --workspace

# Does the code compile?
# --allow warnings to prevent littering the terminal. Most of the warnings are opinionated, slowing down the development process.
cargo clippy --workspace -- --allow warnings
