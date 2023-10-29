#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <mutex>

const int N = 100;

std::vector<bool> is_prime(N + 1, true);
std::mutex mtx;

void mark_multiples(int p) {
    for (int i = p * p; i <= N; i += p) {
        {
            std::lock_guard<std::mutex> lock(mtx);
            is_prime[i] = false;
        }
    }
}

void sieve_of_eratosthenes(int start, int end) {
    for (int p = 2; p * p <= end; p++) {
        if (is_prime[p]) {
            mark_multiples(p);
        }
    }
}

void runThreads(int threadsNumber)
{
  std::vector<std::thread> threads;
  for (int i = 0; i < threadsNumber; i++) {
      int start = (i * N) / threadsNumber;
      int end = ((i + 1) * N) / threadsNumber - 1;
      threads.emplace_back(sieve_of_eratosthenes, start, end);
  }

  for (auto &th : threads)
  {
    th.join();
  }
}

int main()
{
  // int threadsNumber = std::thread::hardware_concurrency();
  int threadsNumber = 10;

  auto start = std::chrono::high_resolution_clock::now();
  runThreads(threadsNumber);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Time taken by threads: "
            << duration.count() << " microseconds" << std::endl;

  // print primes
  for (int p = 2; p <= N; p++) {
      if (is_prime[p]) {
          std::cout << p << " ";
      }
  }

  return 0;
}
