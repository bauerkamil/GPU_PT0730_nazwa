#include <iostream>
#include <vector>
#include <chrono>

// Kernel function for the sieve
__global__ void sieve_of_eratosthenes(bool* is_prime, int chunk_size, int threads_number, int N)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int j, j1, j2;
  int start = index * chunk_size;
  int end = (index == threads_number - 1) ? N : (index + 1) * chunk_size - 1;

  for (int p = 2; p * p <= end; p++)
  {
      j1 = p*p;
      j2 = (start + p - 1) / p * p;
      if (j1 > j2) {
        j = j1;
      } else {
        j = j2;
      }
      for (; j <= end; j += p)
      {
        is_prime[j] = false;
      }
  }
}

void print_primes(bool *is_prime, int N)
{
  for (int i = 2; i <= N; i++)
  {
    if (is_prime[i])
    {
      std::cout << i << " ";
    }
  }
  std::cout << std::endl;
}

void initialize_array(bool* is_prime, int N)
{
  for (int i = 0; i <= N; i++)
  {
    is_prime[i] = true;
  }
}

void run_threads(int threads_number, int N)
{
  bool *is_prime;

  // Allocate Unified Memory – accessible from CPU or GPU
  cudaMallocManaged(&is_prime, (N + 1) * sizeof(bool));

  initialize_array(is_prime, N);

  int chunk_size = N / threads_number;
  sieve_of_eratosthenes<<<28, threads_number / 28>>>(is_prime, chunk_size, threads_number, N);
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // print_primes(is_prime, N);

   // Free memory
  cudaFree(is_prime);
}

int main()
{
  
  const int N = 84000000;

  int threads_number = 3584;
  std::cout << "Threads: " << threads_number << std::endl;

  auto start = std::chrono::high_resolution_clock::now();
  run_threads(threads_number, N);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Time taken by threads: "
            << duration.count() << " microseconds" << std::endl;
}