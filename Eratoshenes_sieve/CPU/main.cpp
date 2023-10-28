#include <iostream>
#include <thread>
#include <vector>
#include <chrono>

// A dummy function
void foo(int params, int params2)
{
  for (int i = 0; i < params; i++)
  {
    std::cout << "Thread using function"
                 " pointer as callable\n";
  }
}

void runThreads(int threadsNumber)
{
  int fooParams = 3;
  int fooParams2 = 2;

  std::vector<std::thread> threads;
  for (int i = 0; i < threadsNumber; ++i)
  {
    threads.push_back(std::thread(foo, fooParams, fooParams2));
  }

  for (auto &th : threads)
  {
    th.join();
  }
}

int main()
{
  int threadsNumber = 10;

  auto start = std::chrono::high_resolution_clock::now();
  runThreads(threadsNumber);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  std::cout << "Time taken by threads: "
            << duration.count() << " nanoseconds" << std::endl;

  return 0;
}
