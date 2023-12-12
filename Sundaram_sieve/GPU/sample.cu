#include <stdio.h>
#include <iostream>
#include <chrono>
#include <fstream>

const uint64_t MAX_STRIDE = 256000000;
int BLOCK_SIZE = 256;

// CUDA kernel to mark non-prime numbers
__global__ void sieveKernel(int *numbers, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x + 1;
    uint64_t stride = blockDim.x * gridDim.x;   
      
    for (uint64_t i = index; i <= n; i += stride)
    {
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
    // calculate number of blocks
    uint64_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint64_t stride = BLOCK_SIZE * num_blocks;

    if (stride > MAX_STRIDE)
    {
        num_blocks = MAX_STRIDE / BLOCK_SIZE;
    }

    // Launch kernel
    sieveKernel<<<num_blocks, BLOCK_SIZE>>>(d_numbers, n);

    // Copy result back from device to host
    cudaMemcpy(numbers, d_numbers, (n + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_numbers);
}

int main(int argc, char *argv[]) {
    int n = 10000;

    for (int i = 1; i < argc; ++i) {
        int integerValue = std::atoi(argv[i]);

        if (integerValue == 0 && argv[i][0] != '0') {
            std::cerr << "Invalid integer: " << argv[i] << std::endl;
            
        } else{
            if(i==1){
                n = integerValue;
            }else{
                BLOCK_SIZE = integerValue;
            }
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    n /= 2;
    // Initialize array with numbers from 1 to n
    int *numbers = new int[n + 1];
    for (int i = 0; i <= n; i++) {
        numbers[i] = 1;
    }

    // Perform Sieve of Sundaram on GPU
    sieveOfSundaramGPU(numbers, n);

    int count = 1;
    printf("2 ");
    for (int i = 1; i < n; i++) {
        if (numbers[i] != 0) {
            // printf("%d ", 2 * i + 1);
            count++;
        }
    }
    printf("\nCount: %d", count);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time taken by threads: "
              << duration.count() << " microseconds" << std::endl;

    delete[] numbers;
    std::ifstream inFile("outputSundaGPU.csv", std::ios::app);
    inFile.seekg(0, std::ios::end);
    std::streampos fileSize = inFile.tellg();

    bool isEmpty = (fileSize == 0);
    inFile.close();

    std::ofstream outFile("outputSundaGPU.csv", std::ios::app);
    if (!outFile) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }
    if (isEmpty) {
         outFile << "size;time\n";
    }
    outFile << n * 2 << ";" << duration.count() << "\n";

    return 0;
}