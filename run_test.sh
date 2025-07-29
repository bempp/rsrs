#!/bin/bash

# Set environment variables
export OPENBLAS_NUM_THREADS=1

# Run Cargo
cargo run --example "$@" --release
