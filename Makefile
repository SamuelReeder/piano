# Variables
IMAGE_NAME = pytorch-container
REMOTE_IMAGE = samuelreeder/piano:$(IMAGE_NAME)
DOCKERFILE_PATH = .
WORKDIR = /workspace
HOST_PORT = 8888

# Default target
.PHONY: all
all: setup build run

# Setup target for a new machine
.PHONY: setup
setup:
	@which docker || (echo "Docker is not installed. Please install Docker before proceeding." && exit 1)
	docker pull $(REMOTE_IMAGE) || echo "Remote image not found, proceeding to build locally..."

# Build the Docker image
.PHONY: build
build:
	@echo "Building Docker image..."
	docker build -t $(IMAGE_NAME) -t $(REMOTE_IMAGE) $(DOCKERFILE_PATH)

# Run the Docker container interactively with GPU support
.PHONY: run
run:
	@echo "Running Docker container..."
	docker run -it --gpus all -v $(shell pwd):$(WORKDIR) $(IMAGE_NAME) /bin/bash

# Clean up stopped containers and dangling images
.PHONY: clean
clean:
	@echo "Cleaning up stopped containers and unused images..."
	docker container prune -f
	docker image prune -f

# Stop and remove the Docker container
.PHONY: stop
stop:
	@echo "Stopping and removing the Docker container..."
	docker stop $(IMAGE_NAME) || true
	docker rm $(IMAGE_NAME) || true

# Remove the Docker image
.PHONY: rmi
rmi:
	@echo "Removing the Docker image..."
	docker rmi $(IMAGE_NAME)
	docker rmi $(REMOTE_IMAGE)
