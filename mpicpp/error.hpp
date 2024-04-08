#pragma once
#ifndef MPI_ERROR_HPP
#define MPI_ERROR_HPP

#include <mpi.h>
#include <stdexcept>

namespace mpi
{

#ifdef MPICPP_ERROR_EXCEPTION
class mpi_error : public std::exception
{
    std::string m_error_string;

  public:
    mpi_error(int errorcode)
    {
        char error_string[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(errorcode, error_string, &length);
        m_error_string = std::string(error_string, length);
    }
    mpi_error(const char *error_msg) : m_error_string(error_msg) {}
    const char *what() const noexcept override { return m_error_string.c_str(); }
};

inline void CHECK_MPI(int error)
{
    if (error != MPI_SUCCESS)
    {
        throw mpi_error(error);
    }
}
#elif defined(MPICPP_ERROR_ABORT)
inline void CHECK_MPI(int error)
{
    if (error != MPI_SUCCESS)
    {
        char error_string[MPI_MAX_ERROR_STRING];
        int length;
        MPI_Error_string(error, error_string, &length);
        error_string[length] = '\0';
        std::cerr << error_string << std::endl;
        MPI_Abort(MPI_COMM_WORLD, error);
    }
}
#else  // do nothing
inline void CHECK_MPI(int error) {}
#endif // MPICPP_ERROR_EXCEPTION

} // end namespace mpi

#endif // MPI_ERROR_HPP