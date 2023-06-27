# CUFFT Benchmark

This is a CUDA program that benchmarks the performance of the CUFFT library for computing FFTs on NVIDIA GPUs. The program generates random input data and measures the time it takes to compute the FFT using CUFFT. The FFT sizes are chosen to be the ones predominantly used by the COMPACT project. 

## Prerequisites

- Nvidia GPU with CUDA drivers that support CUDA 12.0

- Docker 

## Usage

1. Clone the repository
    ```bash
    git clone https://github.com/example/cufft-benchmark.git
    ```
2. `cd` inside the repository
    ```bash
    cd cufft-benchmark
    ```

2. Build the Docker image:
    ```bash
    docker build -t cufft-benchmark .
    ```
3. Run the Docker container:
    ```bash
    docker run --gpus device=<device_id> -it cufft-benchmark
    ```

    This will start the container and will automatically run the benchmark. <device_id> is the id of the GPU. If you have only one GPU on the device, you can also say `--gpus all` instead. 

4. The benchmark output will be printed on the screen. Copy it over and add it to your tender document. 


