#include <iostream>
#include <vector>
#include <thread>
#include <chrono>

const int N = 100000;
const int K = (N - 2) / 2 + 1;
std::mutex mutex;
std::vector<bool> is_prime(K, true);

void sieve_of_sundaram(int start, int end) {
    for (long i = start; i <= end; i++) {
        long j = i;
        while (i + j + 2 * i * j <= K && i + j + 2 * i * j > 0) {
            int index = i + j + 2 * i * j;
            mutex.lock();
            is_prime[index] = false;
            mutex.unlock;
            j += 1;
        }
    }
}
//TODO delete this

void run_threads(int threads_number) {
    int chunk_size = K / threads_number;
    std::vector<std::thread> threads;
    for (int i = 0; i < threads_number; i++) {
        int start = i * chunk_size + 1;
        int end = (i == threads_number - 1) ? K : (i + 1) * chunk_size;
        threads.emplace_back(sieve_of_sundaram, start, end);
    }

    for (auto &th: threads) {
        th.join();
    }
}

int main() {
    int threads_number = std::thread::hardware_concurrency();
    std::cout << "Target num: " << N << std::endl;
    std::cout << "Threads: " << threads_number << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    run_threads(threads_number);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time taken by threads: "
              << duration.count() << " microseconds" << std::endl;

//    // print primes
//    for (int i = 1; 2 * i + 1 <= N; i++) {
//        if (is_prime[i]) {
//            std::cout << 2 * i + 1 << " ";
//        }
//    }

    return 0;
}