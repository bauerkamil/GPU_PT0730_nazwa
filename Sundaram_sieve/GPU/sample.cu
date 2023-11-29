#include <stdio.h>

// CUDA kernel to mark non-prime numbers
__global__ void sieveKernel(int *numbers, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
    
    if (i <= n) {
        for (int j = i; j <= (n - i) / (2 * i + 1); j++) {
            numbers[i + j + 2 * i * j] = 0; // Mark non-prime numbers as 0
        }
    }
}

// Host function to initialize and launch CUDA kernel
void sieveOfSundaramGPU(int *numbers, int n) {
    int *d_numbers;

    // Allocate device memory
    cudaMalloc((void**)&d_numbers, (n + 1) * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_numbers, numbers, (n + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // Set up grid and block dimensions
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch kernel
    sieveKernel<<<gridSize, blockSize>>>(d_numbers, n);

    // Copy result back from device to host
    cudaMemcpy(numbers, d_numbers, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_numbers);
}

int main() {
    int n = 500000;

    // Initialize array with numbers from 1 to n
    int *numbers = new int[n + 1];
    for (int i = 0; i <= n; i++) {
        numbers[i] = 1;
    }

    // Perform Sieve of Sundaram on GPU
    sieveOfSundaramGPU(numbers, n);

    // int count = 1;

    // Display prime numbers
    // printf("2 ");
    // for (int i = 1; i < n; i++) {
    //     if (numbers[i] != 0) {
    //         printf("%d ", 2 * i + 1);
    //         count++;
    //     }
    // }
    // printf("\nCount: %d", count);

    delete[] numbers;

    return 0;
}