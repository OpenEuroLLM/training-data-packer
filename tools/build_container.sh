#!/usr/bin/env bash


container build --arch arm64 --arch amd64 -t ghcr.io/openeurollm/training-data-packer:latest
container image push ghcr.io/openeurollm/training-data-packer:latest
