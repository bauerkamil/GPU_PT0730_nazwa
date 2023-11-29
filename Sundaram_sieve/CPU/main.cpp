#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

const int N = 100000;
const int K = (N - 2) / 2;
// std::vector<bool> is_prime(K, true);
bool *is_prime;

void sieve_of_sundaram(int start, int end)
{
    for (long i = start; i <= end; i++)
    {
        if (is_prime[i])
        {
            long j = i;
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

void print_primes()
{
    int count = 1;
    std::cout << "Primes: 2 ";
    for (int p = 1; p <= K; p++)
    {
        if (is_prime[p])
        {
            std::cout << 2 * p + 1 << " ";
            count++;
        }
    }
    std::cout << std::endl
              << "Count: " << count << std::endl;
}

void run_threads(int threads_number, bool print_results = false)
{

    is_prime = new bool[K + 1];
    for (int i = 0; i <= K; i++)
    {
        is_prime[i] = true;
    }

    int chunk_size = K / threads_number;
    std::vector<std::thread> threads;
    for (int i = 0; i < threads_number; i++)
    {
        int start = i * chunk_size + 1;
        int end = (i == threads_number - 1) ? K : (i + 1) * chunk_size;
        threads.emplace_back(sieve_of_sundaram, start, end);
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
    std::cout << "Target num: " << N << std::endl;
    std::cout << "Threads: " << threads_number << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    run_threads(threads_number);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time taken by threads: "
              << duration.count() << " microseconds" << std::endl;

    //     print_primes();

    return 0;
}