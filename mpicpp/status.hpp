#pragma once
#ifndef MPI_STATUS_HPP
#define MPI_STATUS_HPP

#include "error.hpp"
#include "types.hpp"

namespace mpi
{

class status
{
  private:
    MPI_Status m_status;

  public:
    status(const MPI_Status &status) : m_status(status) {}
    status() : m_status() {}
    int source() const { return m_status.MPI_SOURCE; }
    int tag() const { return m_status.MPI_TAG; }
    int error() const { return m_status.MPI_ERROR; }
    template <typename T>
    int get_count() const
    {
        int count;
        CHECK_MPI(MPI_Get_count(&m_status, mpi_type<T>(), &count));
        return count;
    }
    MPI_Status *ptr() { return &m_status; }
};

} // end namespace mpi

#endif // MPI_STATUS_HPP