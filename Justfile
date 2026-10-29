_default:
    @just --choose

BUILD_DIR := "build"
BUILD_ARGS := "-Dbuildtype=debugoptimized"

setup:
    @out=$(meson setup {{BUILD_DIR}} {{BUILD_ARGS}} 2>&1) || { printf '%s\n' "$out"; exit 1; }

build: setup
    @out=$(NIX_ENFORCE_NO_NATIVE=0 ninja -C {{BUILD_DIR}} 2>&1) || { printf '%s\n' "$out"; exit 1; }

run: build
    @./{{BUILD_DIR}}/nn-out
    @read
