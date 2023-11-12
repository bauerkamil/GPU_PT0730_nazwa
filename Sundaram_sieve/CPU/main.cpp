#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

const int N = 100000;
const int K = (N - 2) / 2;
std::vector<bool> is_prime(K, true);

void sieve_of_sundaram(int start, int end)
{
    for (long i = start; i <= end; i++)
    {
        if (is_prime[i]) {
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

void run_threads(int threads_number)
{
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
}

void print_primes()
{
    for (int p = 0; p <= K; p++)
    {
        if (is_prime[p])
        {
            std::cout << 2 * p + 1 << " ";
        }
    }
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