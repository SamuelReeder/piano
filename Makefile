# Variables
IMAGE_NAME = pytorch-container
DOCKERFILE_PATH = .
WORKDIR = /workspace
HOST_PORT = 8888

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
