#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>
#include <chrono>

const uint64_t MAX_STRIDE = 256000000;
int BLOCK_SIZE = 256;
bool* sieve_buffer_host = nullptr;

__global__ void sieve_kernel(uint64_t max, bool *sieve_buffer)
{
    uint64_t index = blockIdx.x * blockDim.x + threadIdx.x; // index = range of 0 up to MAX_STRIDE
    uint64_t stride = blockDim.x * gridDim.x;               // stride is a number of total threads in grid =>  
                                                            // Grid contains X number of Blocks which contains Y number of Threads 
    for (uint64_t i = index; i <= max; i += stride)
    {
        if(sieve_buffer[i]) {
            uint64_t j = i;
            uint64_t val = i + j + 2 * i * j;
            while (val <= max && val >= i + j)
            {
                uint64_t index2 = i + j + 2 * i * j;
                sieve_buffer[index2] = false;
                j += 1;
                val = i + j + 2 * i * j;
            }
        }
    }
}

void sieve_of_sundaram_gpu_followup(int max)
{
    bool *sieve_buffer_device = nullptr;
    // uint64_t *seed_primes_device = nullptr;

    // allocate on cpu and gpu array of bool and set default true
    sieve_buffer_host = (bool*) malloc(max * sizeof(bool));
    cudaMalloc(&sieve_buffer_device, max * sizeof(bool));
    std::memset(sieve_buffer_host, true, max * sizeof(bool));
    cudaMemcpy(sieve_buffer_device, sieve_buffer_host, max * sizeof(bool), cudaMemcpyHostToDevice);

    // allocate on cpu and gpu for seed_primes and copy seed_primes to gpu
    // uint64_t seed_primes_size = seed_primes.size();
    // uint64_t *seed_primes_host = (uint64_t *)malloc(seed_primes_size * sizeof(uint64_t));
    // memcpy(seed_primes_host, seed_primes.data(), seed_primes_size * sizeof(uint64_t));
    // cudaMalloc(&seed_primes_device, seed_primes_size * sizeof(uint64_t));
    // cudaMemcpy(seed_primes_device, seed_primes_host, seed_primes_size * sizeof(uint64_t), cudaMemcpyHostToDevice);

    // calculate number of blocks
    uint64_t num_blocks = (max + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uint64_t stride = BLOCK_SIZE * num_blocks;

    if (stride > MAX_STRIDE)
    {
        num_blocks = MAX_STRIDE / BLOCK_SIZE;
    }

    // run on GPU
    sieve_kernel<<<num_blocks, BLOCK_SIZE>>>(max, sieve_buffer_device);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    cudaMemcpy(sieve_buffer_host, sieve_buffer_device, max * sizeof(bool), cudaMemcpyDeviceToHost);

    // free(seed_primes_host);
    cudaFree(sieve_buffer_device);
    // cudaFree(seed_primes_device);
}

void check_primes(int target)
{

    uint64_t count = 1;
    // std::cout << 2 << " ";
    for (int p = 1; p <= target; p++)
    {
        if (sieve_buffer_host[p])
        {
            count++;
            // std::cout << 2 * p + 1 << " ";
        }
    }
    std::cout << "Found " << count << " prime numbers" << std::endl; 
}

int main(int argc, char *argv[])
{
    int target = 400;

    for (int i = 1; i < argc; ++i) {
        int integerValue = std::atoi(argv[i]);

        if (integerValue == 0 && argv[i][0] != '0') {
            std::cerr << "Invalid integer: " << argv[i] << std::endl;
            
        }else{
            if(i==1){
                target = integerValue;
            }
            if(i==2){
                BLOCK_SIZE = integerValue;
            }
        }
    }

    auto start = std::chrono::high_resolution_clock::now();
    int k = (target - 2) / 2;
    // std::vector<uint64_t> seed_primes = sieve_of_sundaram_cpu(sqrt);
    sieve_of_sundaram_gpu_followup(k+1);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time taken by threads: "
              << duration.count() << " microseconds" << std::endl;

    check_primes(k);
}