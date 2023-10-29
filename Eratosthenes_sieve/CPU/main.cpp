#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <mutex>

const int N = 100000;

std::vector<bool> is_prime(N + 1, true);
std::mutex mtx;

void mark_multiples(int p)
{
    for (int i = p * p; i <= N; i += p)
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            is_prime[i] = false;
            std::lock_guard<std::mutex> unlock(mtx);
        }
    }
}

void sieve_of_eratosthenes(int start, int end)
{
    for (int p = 2; p * p <= end; p++)
    {
        if (is_prime[p])
        {
            mark_multiples(p);
        }
    }
}

void run_threads(int threads_number)
{
    std::vector<std::thread> threads;
    for (int i = 0; i < threads_number; i++)
    {
        int start = (i * N) / threads_number;
        int end = ((i + 1) * N) / threads_number - 1;
        threads.emplace_back(sieve_of_eratosthenes, start, end);
    }

    for (auto &th : threads)
    {
        th.join();
    }
}

int main()
{
    // int threads_number = std::thread::hardware_concurrency();
    int threads_number = 10;

    std::cout << "Threads: " << threads_number << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    run_threads(threads_number);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time taken by threads: "
              << duration.count() << " microseconds" << std::endl;

    // print primes
    // for (int p = 2; p <= N; p++)
    // {
    //     if (is_prime[p])
    //     {
    //         std::cout << p << " ";
    //     }
    // }

    return 0;
}
