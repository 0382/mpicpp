#pragma once
#ifndef MPI_BASE_HPP
#define MPI_BASE_HPP

#include "environment.hpp"
#include "request.hpp"
#include "status.hpp"
#include "types.hpp"
#include <mutex>
#include <stdexcept>
#include <vector>

namespace mpi
{

class communicator
{
  public:
#ifdef MPICPP_USE_EXCEPTION
    void check(int error) const
    {
        if (error != MPI_SUCCESS)
        {
            throw mpi_error(error);
        }
    }
#else
    void check(int error) const
    {
        if (error != MPI_SUCCESS)
        {
            char error_string[MPI_MAX_ERROR_STRING];
            int length;
            MPI_Error_string(error, error_string, &length);
            error_string[length] = '\0';
            std::cerr << error_string << std::endl;
            this->abort(error);
        }
    }
#endif

    communicator(MPI_Comm comm) : m_comm(comm) {}
    int rank() const
    {
        static thread_local int local_rank = -1;
        static std::once_flag flag;
        std::call_once(flag, [this]() { check(MPI_Comm_rank(m_comm, &local_rank)); });
        return local_rank;
    }
    int size() const
    {
        static thread_local int local_size = -1;
        static std::once_flag flag;
        std::call_once(flag, [this]() { check(MPI_Comm_size(m_comm, &local_size)); });
        return local_size;
    }

    void barrier() { check(MPI_Barrier(m_comm)); }
    void abort(int errorcode) const { MPI_Abort(m_comm, errorcode); }

    // ----- broadcast -----

    template <typename T>
    void broadcast(T *buf, std::size_t count, int root) const
    {
        check_type<T>();
        check(MPI_Bcast(buf, count, mpi_type<T>(), root, m_comm));
    }

    template <typename T>
    void broadcast(T &buf, int root) const
    {
        broadcast<T>(&buf, 1, root);
    }

    template <typename T>
    void broadcast(std::vector<T> &data, int root) const
    {
        std::size_t size;
        if (rank() == root)
        {
            size = data.size();
        }
        broadcast<std::size_t>(size, root);
        if (rank() != root)
        {
            data.resize(size);
        }
        broadcast<T>(data.data(), data.size(), root);
    }

    void broadcast(std::string &str, int root) const
    {
        std::size_t size;
        if (rank() == root)
        {
            size = str.size();
        }
        broadcast<std::size_t>(size, root);
        if (rank() != root)
        {
            str.resize(size);
        }
        broadcast<char>(str.data(), str.size(), root);
    }

    // ----- send -----

    template <typename T>
    void send(const T *buf, std::size_t count, int dest, int tag) const
    {
        check_type<T>();
        check(MPI_Send(buf, count, mpi_type<T>(), dest, tag, m_comm));
    }

    template <typename T>
    void send(const T buf, int dest, int tag) const
    {
        send<T>(&buf, 1, dest, tag);
    }

    template <typename T>
    void send(const std::vector<T> &data, int dest, int tag) const
    {
        send<std::size_t>(data.size(), dest, tag);
        send<T>(data.data(), data.size(), dest, tag);
    }

    void send(const std::string &str, int dest, int tag) const
    {
        send<std::size_t>(str.size(), dest, tag);
        send<char>(str.data(), str.size(), dest, tag);
    }

    // ----- recv -----

    template <typename T>
    void recv(T *buf, std::size_t count, int src, int tag) const
    {
        check_type<T>();
        check(MPI_Recv(buf, count, mpi_type<T>(), src, tag, m_comm, MPI_STATUS_IGNORE));
    }

    template <typename T>
    void recv(T &buf, int src, int tag) const
    {
        return recv<T>(&buf, 1, src, tag);
    }

    template <typename T>
    void recv(std::vector<T> &data, int src, int tag) const
    {
        std::size_t size;
        recv<std::size_t>(size, src, tag);
        data.resize(size);
        recv<T>(data.data(), data.size(), src, tag);
    }

    void recv(std::string &str, int src, int tag) const
    {
        std::size_t size;
        recv<std::size_t>(size, src, tag);
        str.resize(size);
        recv<char>(str.data(), str.size(), src, tag);
    }

    template <typename T>
    void recv(T *buf, std::size_t count, int src, int tag, status &st) const
    {
        check_type<T>();
        check(MPI_Recv(buf, count, mpi_type<T>(), src, tag, m_comm, st.ptr()));
    }

    template <typename T>
    void recv(T &buf, int src, int tag, status &st) const
    {
        recv<T>(&buf, 1, src, tag, st);
    }

    template <typename T>
    void recv(std::vector<T> &data, int src, int tag, status &st) const
    {
        std::size_t size;
        recv<std::size_t>(size, src, tag, st);
        data.resize(size);
        recv<T>(data.data(), data.size(), src, tag, st);
    }

    void recv(std::string &str, int src, int tag, status &st) const
    {
        std::size_t size;
        recv<std::size_t>(size, src, tag, st);
        str.resize(size);
        recv<char>(str.data(), str.size(), src, tag, st);
    }

    // ----- isend -----

    template <typename T>
    request isend(const T *buf, std::size_t count, int dest, int tag) const
    {
        check_type<T>();
        MPI_Request req;
        check(MPI_Isend(buf, count, mpi_type<T>(), dest, tag, m_comm, &req));
        return request{req};
    }

    template <typename T>
    request isend(const T buf, int dest, int tag) const
    {
        return isend<T>(&buf, 1, dest, tag);
    }

    // ----- irecv -----

    template <typename T>
    request irecv(T *buf, std::size_t count, int src, int tag) const
    {
        check_type<T>();
        MPI_Request req;
        check(MPI_Irecv(buf, count, mpi_type<T>(), src, tag, m_comm, &req));
        return request{req};
    }

    template <typename T>
    request irecv(T &buf, int src, int tag) const
    {
        return irecv<T>(&buf, 1, src, tag);
    }

    // ----- scatter -----

    template <typename T>
    void scatter(const T *send_data, T *recv_data, std::size_t count, int root)
    {
        check_type<T>();
        MPI_Scatter(send_data, count, mpi_type<T>(), recv_data, count, mpi_type, root, m_comm);
    }

    template <typename T>
    void scatter(const T *send_data, T &recv_data, int root)
    {
        scatter<T>(send_data, &recv_data, 1, root);
    }

    // for non-root process, send_data is not needed.
    template <typename T>
    void scatter(T &recv_data, int root)
    {
        scatter<T>(nullptr, &recv_data, 1, root);
    }

    // ----- gather -----

    template <typename T>
    void gather(const T *send_data, T *recv_data, std::size_t count, int root)
    {
        check_type<T>();
        MPI_Gather(send_data, count, mpi_type<T>(), recv_data, count, mpi_type<T>(), root, m_comm);
    }

    template <typename T>
    void gather(const T send_data, T *recv_data, int root)
    {
        gather<T>(&send_data, recv_data, 1, root);
    }

    // for non-root process, recv_data is not needed.
    template <typename T>
    void gather(const T send_data, int root)
    {
        gather<T>(&send_data, nullptr, 1, root);
    }

    // ----- allgather -----

    template <typename T>
    void allgather(const T *send_data, T *recv_data, std::size_t count)
    {
        check_type<T>();
        MPI_Allgather(send_data, count, mpi_type<T>(), recv_data, count, mpi_type, m_comm);
    }

    template <typename T>
    void allgather(const T send_data, T *recv_data)
    {
        allgather<T>(&send_data, recv_data, 1);
    }

    // ----- reduce -----

    template <typename T>
    void reduce(const T *send_data, T *recv_data, std::size_t count, MPI_Op op, int root)
    {
        check_type<T>();
        ;
        MPI_Reduce(send_data, recv_data, count, mpi_type<T>(), op, root, m_comm);
    }

    template <typename T>
    void reduce(const T send_data, T &recv_data, MPI_Op op, int root)
    {
        reduce<T>(&send_data, &recv_data, 1, op, root);
    }

    template <typename T>
    void reduce(const T send_data, MPI_Op op, int root)
    {
        reduce<T>(&send_data, nullptr, 1, op, root);
    }

    // ----- allreduce -----

    template <typename T>
    void allreduce(const T *send_data, T *recv_data, std::size_t count, MPI_Op op)
    {
        check_type<T>();
        MPI_Allreduce(send_data, recv_data, count, mpi_type<T>(), op, m_comm);
    }

    template <typename T>
    void allreduce(const T send_data, T &recv_data, MPI_Op op)
    {
        allreduce<T>(&send_data, &recv_data, 1, op);
    }

    // ----- alltoall -----

    template <typename T>
    void alltoall(const T *send_data, int send_count, T *recv_data, int recv_count)
    {
        check_type<T>();
        MPI_Alltoall(send_data, send_count, mpi_type<T>(), recv_data, recv_count, mpi_type<T>(), m_comm);
    }

    // 只发送一个的情况
    template <typename T>
    void alltoall(const T *send_data, T *recv_data)
    {
        check_type<T>();
        MPI_Alltoall(send_data, 1, mpi_type<T>(), recv_data, 1, mpi_type<T>(), m_comm);
    }

  private:
    MPI_Comm m_comm;
};

inline communicator world(MPI_COMM_WORLD);

} // end namespace mpi

#endif // MPI_BASE_HPP