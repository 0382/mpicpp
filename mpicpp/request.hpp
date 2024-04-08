#pragma once
#ifndef MPI_REQUEST_HPP
#define MPI_REQUEST_HPP

#include "error.hpp"
#include "status.hpp"
#include <vector>

namespace mpi
{

class request
{
  private:
    MPI_Request m_request;

  public:
    request() : m_request(MPI_REQUEST_NULL) {}
    request(MPI_Request request) : m_request(request) {}
    bool valid() const { return m_request != MPI_REQUEST_NULL; }
    void wait(status &st)
    {
        if (!valid())
            return;
        CHECK_MPI(MPI_Wait(&m_request, st.ptr()));
    }
    void wait()
    {
        if (!valid())
            return;
        CHECK_MPI(MPI_Wait(&m_request, MPI_STATUS_IGNORE));
    }
    bool test(status &st)
    {
        if (!valid())
            return true; // completed
        int flag;
        CHECK_MPI(MPI_Test(&m_request, &flag, st.ptr()));
        return flag;
    }
    bool test()
    {
        if (!valid())
            return true; // completed
        int flag;
        CHECK_MPI(MPI_Test(&m_request, &flag, MPI_STATUS_IGNORE));
        return flag;
    }
    void cancel()
    {
        if (!valid())
            return;
        CHECK_MPI(MPI_Cancel(&m_request));
        CHECK_MPI(MPI_Request_free(&m_request));
    }
    ~request() { wait(); }
};

inline void wait_all(std::size_t count, request *requests, status *statuses)
{
    MPI_Request *req = reinterpret_cast<MPI_Request *>(requests);
    MPI_Status *st = reinterpret_cast<MPI_Status *>(statuses);
    CHECK_MPI(MPI_Waitall(count, req, st));
}

inline int wait_any(std::size_t count, request *requests, status &st)
{
    int index = {};
    MPI_Request *req = reinterpret_cast<MPI_Request *>(requests);
    CHECK_MPI(MPI_Waitany(count, req, &index, st.ptr()));
    return index;
}

inline std::vector<int> wait_some(std::size_t count, request *requests, status *statuses)
{
    int outcount = {};
    std::vector<int> indices(count);
    MPI_Request *req = reinterpret_cast<MPI_Request *>(requests);
    MPI_Status *st = reinterpret_cast<MPI_Status *>(statuses);
    CHECK_MPI(MPI_Waitsome(count, req, &outcount, indices.data(), st));
    indices.resize(outcount);
    return indices;
}

} // end namespace mpi

#endif // MPI_REQUEST_HPP