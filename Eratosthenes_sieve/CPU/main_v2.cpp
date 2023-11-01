#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cmath>

const int N = 200;
std::vector<bool> primes(N, true);

void markMultiples(std::vector<bool>& primes, int start, int n, int factor) {
    for (int i = start; i < n; i += factor) {
        primes[i] = false;
    }
}

void sieveOfEratosthenes(int n) {

    int sqrtN = std::sqrt(n);
    std::vector<std::thread> threads;

    for (int p = 2; p <= sqrtN; ++p) {
        if (primes[p]) {
            threads.push_back(std::thread(markMultiples, std::ref(primes), p * p, n, p));
        }
    }

    for (std::thread& thread : threads) {
        thread.join();
    }
}

void runThreads(int threadsNumber)
{
    sieveOfEratosthenes(N + 1);
}

int main()
{
  int threadsNumber = 10;

  auto start = std::chrono::high_resolution_clock::now();
  runThreads(threadsNumber);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Time taken by threads: "
            << duration.count() << " microseconds" << std::endl;

  // print primes
  for (int p = 2; p <= N; p++) {
      if (primes[p]) {
          std::cout << p << " ";
      }
  }

  return 0;
}
