#pragma once
#ifndef MPI_ENVIRONMENT_HPP
#define MPI_ENVIRONMENT_HPP

#include "error.hpp"

namespace mpi
{

class environment
{
  public:
    environment(int argc, char **argv) { CHECK_MPI(MPI_Init(&argc, &argv)); }
    ~environment() { CHECK_MPI(MPI_Finalize()); }
    static bool initialized()
    {
        int flag;
        CHECK_MPI(MPI_Initialized(&flag));
        return flag;
    }
    static bool finalized()
    {
        int flag;
        CHECK_MPI(MPI_Finalized(&flag));
        return flag;
    }
};

} // end namespace mpi

#endif // MPI_ENVIRONMENT_HPP