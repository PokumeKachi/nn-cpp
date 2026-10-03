_default:
    @just --choose

BUILD_DIR := "build"
BUILD_ARGS := "-Dbuildtype=release"

setup:
    meson setup {{BUILD_DIR}} {{BUILD_ARGS}}

build: setup
    ninja -C {{BUILD_DIR}}

run: build
    ./{{BUILD_DIR}}/nn-out
    @read
