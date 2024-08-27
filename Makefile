IMAGE_NAME = pytorch-container
DOCKERFILE_PATH = .
WORKDIR = /workspace
HOST_PORT= 8888 

# Default target
.PHONY: all
all: build run

# Build the Docker image
.PHONY: build
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) $(DOCKERFILE_PATH)

# Run the Docker container interactively with GPU support
.PHONY: run
run:
	@echo "Running Docker container..."
	docker run -it --gpus all -v $(shell pwd):$(WORKDIR) $(IMAGE_NAME) /bin/bash
