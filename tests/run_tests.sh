#!/usr/bin/env bash
# builds and runs every test in this directory, exits non-zero if any test fails
set -e

cd "$(dirname "$0")"
build_dir=$(mktemp -d)
trap 'rm -rf "$build_dir"' EXIT

for src in test_*.cpp; do
    name="${src%.cpp}"
    echo "=== $name ==="
    clang++ -std=c++23 -O0 -I.. "$src" -o "$build_dir/$name"
    "$build_dir/$name"
    echo
done

echo "all tests passed"
