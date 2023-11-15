#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

// Kernel function for the sieve
__global__ void sieve_of_sundaram(bool* is_prime, int chunk_size, int threads_number, int K)
{
    // int index = blockIdx.x * blockDim.x + threadIdx.x;
    // int stride = blockDim.x * gridDim.x;
    int index = threadIdx.x;
    int start = index * chunk_size + 1;
    int end = (index == threads_number - 1) ? K : (index + 1) * chunk_size;
    for (int i = start; i <= end; i++)
    {
        if(is_prime[i]) {
            int j = i;
            long val = i + j + 2 * i * j;
            while (val <= K && val > 0)
            {
                int index = i + j + 2 * i * j;
                is_prime[index] = false;
                j += 1;
                val = i + j + 2 * i * j;
            }
        }
    }
}

void print_primes(bool *is_prime, int K)
{
    memset(is_prime, false, (K + 1) * sizeof(bool));
    for (int p = 0; p <= K; p++)
    {
        if (is_prime[p])
        {
            std::cout << 2 * p + 1 << " ";
        }
    }
}

void initialize_array(bool* is_prime, int K)
{
    for (int i = 0; i <= K; i++)
    {
        is_prime[i] = true;
    }
}

void run_threads(int threads_number, int N)
{
    int K = (N - 2) / 2 + 1;
    bool *is_prime;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&is_prime, K * sizeof(bool));

    initialize_array(is_prime, K);
    int chunk_size = K / threads_number;

    sieve_of_sundaram<<<1, threads_number>>>(is_prime, chunk_size, threads_number, K);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // print_primes(is_prime, N);

    // Free memory
    cudaFree(is_prime);
}

int main()
{
    const int N = 2137000000;
    int threads_number = 8192;
    // std::cout << "Target num: " << N << std::endl;
    // std::cout << "Threads: " << threads_number << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    run_threads(threads_number, N);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time taken by threads: "
              << duration.count() << " microseconds" << std::endl;

    return 0;
}