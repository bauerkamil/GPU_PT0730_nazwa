#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

const int N = 1 << 10;
bool *is_prime;

void sieve_of_eratosthenes(int start, int end)
{
    for (int p = 2; p * p <= end; p++)
    {
        if (is_prime[p])
        {
            int j = std::max(p * p, (start + p - 1) / p * p);
            for (; j <= end; j += p)
            {
                is_prime[j] = false;
            }
        }
    }
}

void print_primes()
{
    for (int p = 2; p <= N; p++)
    {
        if (is_prime[p])
        {
            std::cout << p << " ";
        }
    }
}

void run_threads(int threads_number, bool print_results = false)
{
    is_prime = new bool[N + 1];
    for (int i = 0; i < N + 1; i++)
    {
        is_prime[i] = true;
    }

    int chunk_size = N / threads_number;
    std::vector<std::thread> threads;
    for (int i = 0; i < threads_number; i++)
    {
        int start = i * chunk_size;
        int end = (i == threads_number - 1) ? N : (i + 1) * chunk_size - 1;
        threads.emplace_back(sieve_of_eratosthenes, start, end);
    }

    for (auto &th : threads)
    {
        th.join();
    }

    if (print_results)
    {
        print_primes();
    }

    delete[] is_prime;
}

int main()
{
    int threads_number = std::thread::hardware_concurrency();
    // int threads_number = 1;

    std::cout << "Threads: " << threads_number << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    run_threads(threads_number);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time taken by threads: "
              << duration.count() << " microseconds" << std::endl;

    return 0;
}
