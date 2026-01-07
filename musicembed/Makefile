SHELL := /bin/bash

BINARY := musicembed-test
BUILD_DIR := build

.PHONY: build test clean

build:
	@mkdir -p $(BUILD_DIR)
	go build -o $(BUILD_DIR)/$(BINARY) ./

test:
	go run ./test.go

clean:
	rm -rf $(BUILD_DIR)
