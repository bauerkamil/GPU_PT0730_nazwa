#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <fstream>

int N = 1000000;
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
    int count = 0;
    for (int p = 2; p <= N; p++)
    {
        if (is_prime[p])
        {
            // std::cout << p << " ";
            count++;
        }
    }
    std::cout << std::endl
              << "Count: " << count << std::endl;
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

int main(int argc, char *argv[])
{

    for (int i = 1; i < argc; ++i) {
        int integerValue = std::atoi(argv[i]);

        if (integerValue == 0 && argv[i][0] != '0') {
            std::cerr << "Invalid integer: " << argv[i] << std::endl;
            
        } else{
            N = integerValue;
        }
    }

    int threads_number = std::thread::hardware_concurrency();
    // int threads_number = 1;

    std::cout << "Threads: " << threads_number << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    run_threads(threads_number,true);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Time taken by threads: "
              << duration.count() << " microseconds" << std::endl;

    std::ifstream inFile("outputEraCPU.csv", std::ios::app);
    inFile.seekg(0, std::ios::end);
    std::streampos fileSize = inFile.tellg();

    bool isEmpty = (fileSize == 0);
    inFile.close();

    std::ofstream outFile("outputEraCPU.csv", std::ios::app);
    if (!outFile) {
        std::cerr << "Error opening file!" << std::endl;
        return 1;
    }
    if (isEmpty) {
         outFile << "size;time\n";
    } 
    outFile << N << ";" << duration.count() << "\n";

    return 0;
}
