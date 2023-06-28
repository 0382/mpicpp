#include <algorithm>
#include <iostream>
#include <mpi.hpp>
#include <numeric>
#include <thread>

int main(int argc, char *argv[])
{
    mpi::environment env(argc, argv);
    mpi::log_init(mpi::LogLevel::Info);

    using mpi::world;

    std::string s;
    if (world.rank() == 0)
    {
        s = "Hello mpicpp";
        world.send(s, 1, 0);
        mpi::log_info("send `s` from rank ", world.rank());
    }
    else if (world.rank() == 1)
    {
        world.recv(s, 0, 0);
        mpi::log_info("recv `s` to rank ", world.rank(), ", s = ", s);
    }

    std::vector<int> x(world.size(), 0);
    if (world.rank() == 0)
    {
        world.gather(world.rank(), x.data(), 0);
        mpi::log_info("gather to rank ", world.rank(), ", sum of rank = ", std::accumulate(x.begin(), x.end(), 0));
    }
    else
    {
        world.gather(world.rank(), 0);
        mpi::log_info("send to gather from rank ", world.rank());
    }

    int sum;
    // world.allreduce(world.rank(), sum, MPI_SUM); // is also ok.
    world.allreduce(world.rank(), sum, mpi::op::sum());
    mpi::log_info("sum = ", sum);

    if (world.rank() == 0)
    {
        int x = 0;
        // 仅支持仿函数类型
        world.reduce(world.rank(), x, mpi::op::custom<int, std::plus<int>>(true), 0);
        mpi::log_info("x = ", x);
    }
    else
    {
        world.reduce(world.rank(), mpi::op::custom<int, std::plus<int>>(true), 0);
    }

    x.clear();
    x.resize(world.size(), world.rank());
    std::vector<int> y(world.size());
    world.alltoall(x.data(), y.data());
    int next_proc = (world.rank() + 1) % world.size();
    mpi::log_info("recv ", y[next_proc], " from rank ", next_proc);
    return 0;
}
