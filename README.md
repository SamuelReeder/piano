# Pianist AI

A transformer-based model that can generate sequences of piano notes and their respective durations and velocities.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Docker**: Install Docker from [here](https://docs.docker.com/get-docker/).
- **NVIDIA Container Toolkit** (for GPU support): Follow the installation guide [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Getting Started

### Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/SamuelReeder/piano.git
cd piano
```

### Using Docker

This project is fully Dockerized, meaning all dependencies and environment setup is handled via Docker. 

#### Building the Docker Image

To build the Docker image, use the Makefile:

If this is your first time running the project, you should set up the Docker image first by pulling it from the Docker Hub:

```bash
make setup
```

Otherwise, you can build the Docker image directly:

```bash
make build
```

#### Running the Docker Container

To run the Docker container with GPU support, use the following command:

```bash
make run
```

This command will:
- Start the Docker container interactively.
- Mount the current directory to `/workspace` inside the container, so any changes made are reflected on your host system.
- Enable GPU support (assuming you have the NVIDIA Container Toolkit installed).

### Running the Scripts

Once inside the Docker container, you can run the following scripts:

#### Training

To train a model, run the following command:
```bash
python train.py <model_name>
```

This will train a model with the specified name and save it to the `models` directory. Please update the parameters in the `train.py` script to customize the training.

#### Generation

To generate a musical piece and a `.wav` file consisting of the simulated audio, run the following command:

```bash
python generate.py <model_name> <max_tokens> <output_file_name>
```

