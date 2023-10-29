#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <mutex>

const int N = 10000;

std::vector<bool> is_prime((N - 1) / 2 + 1, true);
std::mutex mtx;

void mark_multiples(int p) {
  for (int i = 2 * p; i <= N; i += p) {
    {
      std::lock_guard<std::mutex> lock(mtx);
      is_prime[(i - 1) / 2] = false;
      std::lock_guard<std::mutex> unlock(mtx);
    }
  }
}

void sieve_of_sundaram(int start, int end) {
  for (int i = 1; 2 * i + 1 <= end; i++) {
    if (is_prime[i]) {
      int p = 2 * i + 1;
      mark_multiples(p);
    }
  }
}

void run_threads(int threads_number)
{
  std::vector<std::thread> threads;
  for (int i = 0; i < threads_number; i++) {
    int start = ((i * N) / threads_number + 1) / 2;
    int end = (((i + 1) * N) / threads_number) / 2;
    threads.emplace_back(sieve_of_sundaram, start, end);
  }

  for (auto &th : threads)
  {
  th.join();
  }
}

int main()
{
  int threads_number = std::thread::hardware_concurrency();
  // int threads_number = 10;

  auto start = std::chrono::high_resolution_clock::now();
  run_threads(threads_number);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "Time taken by threads: "
      << duration.count() << " microseconds" << std::endl;

  for (int i = 1; 2 * i + 1 <= N; i++) {
    if (is_prime[i]) {
      std::cout << 2 * i + 1 << " ";
    }
  }

  return 0;
}
