#include <algorithm>
#include <iostream>
#include <mpi.hpp>
#include <numeric>
#include <thread>

int main(int argc, char *argv[])
{
    mpi::enviroment env(argc, argv);
    mpi::log_init(mpi::LogLevel::Info);
    std::vector<int> x;

    std::string gs;
    if (mpi::world.rank() == 0)
    {
        x.resize(10);
        std::iota(x.begin(), x.end(), 1);
        mpi::world.send(x, 1, 0);

        std::string s = "Hello world!";
        mpi::world.send(s, 1, 100);

        gs = "你好世界";
    }
    else if (mpi::world.rank() == 1)
    {
        mpi::world.recv(x, 0, 0);

        std::thread th[10];
        for (int i = 0; i < 10; ++i)
        {
            th[i] = std::thread(
                [i, &x]()
                {
                    for (int j = 0; j < i; ++j)
                    {
                        std::this_thread::sleep_for(std::chrono::seconds(1));
                        mpi::log_info("thread ", i, ", v = ", x[i]);
                    }
                });
        }
        for (int i = 0; i < 10; ++i)
        {
            th[i].join();
        }

        std::string s;
        mpi::world.recv(s, 0, 100);

        mpi::log_info("s = ", s);
    }
    mpi::world.broadcast(gs, 0);
    if (mpi::world.rank() == 1)
    {
        mpi::log_info("gs = ", gs);
    }
    return 0;
}
