.PHONY: all build test lint proto clean help

# Go parameters
GOCMD=go
GOBUILD=$(GOCMD) build
GOTEST=$(GOCMD) test
GOMOD=$(GOCMD) mod
GOVET=$(GOCMD) vet

# Proto parameters
BUF=buf

all: proto build test

## build: Build the project
build:
	$(GOBUILD) -v ./...

## test: Run tests
test:
	$(GOTEST) -v -race ./...

## test-cover: Run tests with coverage
test-cover:
	$(GOTEST) -v -race -coverprofile=coverage.out ./...
	$(GOCMD) tool cover -html=coverage.out -o coverage.html

## lint: Run linter
lint:
	golangci-lint run

## vet: Run go vet
vet:
	$(GOVET) ./...

## proto: Generate protobuf code
proto:
	$(BUF) generate

## proto-lint: Lint protobuf files
proto-lint:
	$(BUF) lint proto

## proto-breaking: Check for breaking changes in proto
proto-breaking:
	$(BUF) breaking proto --against '.git#branch=main'

## tidy: Run go mod tidy
tidy:
	$(GOMOD) tidy

## download: Download dependencies
download:
	$(GOMOD) download

## verify: Verify dependencies
verify:
	$(GOMOD) verify

## clean: Clean build artifacts
clean:
	rm -rf coverage.out coverage.html
	rm -rf gen/

## install-tools: Install development tools
install-tools:
	go install github.com/bufbuild/buf/cmd/buf@latest
	go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
	go install connectrpc.com/connect/cmd/protoc-gen-connect-go@latest
	go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest

## help: Show this help
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@sed -n 's/^##//p' $(MAKEFILE_LIST) | column -t -s ':' | sed -e 's/^/ /'
