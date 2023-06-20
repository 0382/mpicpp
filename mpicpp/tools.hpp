#pragma once
#ifndef MPI_TOOLS_HPP
#define MPI_TOOLS_HPP

#include "logger.hpp"

namespace mpi
{

inline void check_status(MPI_Status st)
{
    thread_local char error_string[2 * MPI_MAX_ERROR_STRING];
    int error;
    MPI_Error_class(st.MPI_ERROR, &error);
    if (error != MPI_SUCCESS)
    {
        int length;
        MPI_Error_string(error, error_string, &length);
        error_string[length] = ':';
        error_string[length + 1] = ' ';
        char *ptr = error_string + length + 2;
        MPI_Error_string(st.MPI_ERROR, ptr, &length);
        log_error_stop(error_string);
    }
}

} // end namespace mpi

#endif // MPI_TOOLS_HPP