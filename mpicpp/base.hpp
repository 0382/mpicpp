#pragma once
#ifndef MPI_BASE_HPP
#define MPI_BASE_HPP

#include "types.hpp"
#include <vector>

namespace mpi
{

class environment
{
  public:
    environment(int argc, char **argv) { MPI_Init(&argc, &argv); }

    ~environment() { MPI_Finalize(); }
};

class communicator
{
  public:
    communicator(MPI_Comm comm) : m_comm(comm) {}
    int rank() const
    {
        static thread_local int rank = -1;
        if (rank == -1)
        {
            MPI_Comm_rank(m_comm, &rank);
        }
        return rank;
    }
    int size() const
    {
        static thread_local int size = -1;
        if (size == -1)
        {
            MPI_Comm_size(m_comm, &size);
        }
        return size;
    }

    template <typename T>
    void send(const T *buf, int count, int dest, int tag) const
    {
        check_type<T>();
        MPI_Send(buf, count, mpi_type_map<T>::type, dest, tag, m_comm);
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

    template <typename T>
    void broadcast(T *buf, int count, int root) const
    {
        check_type<T>();
        MPI_Bcast(buf, count, mpi_type_map<T>::type, root, m_comm);
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

    template <typename T>
    MPI_Status recv(T *buf, int count, int src, int tag) const
    {
        check_type<T>();
        MPI_Status st;
        MPI_Recv(buf, count, mpi_type_map<T>::type, src, tag, m_comm, &st);
        return st;
    }

    template <typename T>
    MPI_Status recv(T &buf, int src, int tag) const
    {
        return recv<T>(&buf, 1, src, tag);
    }

    template <typename T>
    MPI_Status recv(std::vector<T> &data, int src, int tag) const
    {
        std::size_t size;
        MPI_Status st = recv<std::size_t>(size, src, tag);
        if (st.MPI_ERROR != MPI_SUCCESS)
            return st;
        data.resize(size);
        return recv<T>(data.data(), data.size(), src, tag);
    }

    MPI_Status recv(std::string &str, int src, int tag) const
    {
        std::size_t size;
        MPI_Status st = recv<std::size_t>(size, src, tag);
        if (st.MPI_ERROR != MPI_SUCCESS)
            return st;
        str.resize(size);
        return recv<char>(str.data(), str.size(), src, tag);
    }

    template <typename T>
    void scatter(const T *send_data, T *recv_data, int count, int root)
    {
        check_type<T>();
        static constexpr auto mpi_type = mpi_type_map<T>::type;
        MPI_Scatter(send_data, count, mpi_type, recv_data, count, mpi_type, root, m_comm);
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

    template <typename T>
    void gather(const T *send_data, T *recv_data, int count, int root)
    {
        check_type<T>();
        static constexpr auto mpi_type = mpi_type_map<T>::type;
        MPI_Gather(send_data, count, mpi_type, recv_data, count, mpi_type, root, m_comm);
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

    template <typename T>
    void allgather(const T *send_data, T *recv_data, int count)
    {
        check_type<T>();
        static constexpr auto mpi_type = mpi_type_map<T>::type;
        MPI_Allgather(send_data, count, mpi_type, recv_data, count, mpi_type, m_comm);
    }

    template <typename T>
    void allgather(const T send_data, T *recv_data)
    {
        allgather<T>(&send_data, recv_data, 1);
    }

    template <typename T>
    void reduce(const T *send_data, T *recv_data, int count, MPI_Op op, int root)
    {
        check_type<T>();
        static constexpr auto mpi_type = mpi_type_map<T>::type;
        MPI_Reduce(send_data, recv_data, count, mpi_type, op, root, m_comm);
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

    template <typename T>
    void allreduce(const T *send_data, T *recv_data, int count, MPI_Op op)
    {
        check_type<T>();
        static constexpr auto mpi_type = mpi_type_map<T>::type;
        MPI_Allreduce(send_data, recv_data, count, mpi_type, op, m_comm);
    }

    template <typename T>
    void allreduce(const T send_data, T &recv_data, MPI_Op op)
    {
        allreduce<T>(&send_data, &recv_data, 1, op);
    }

    template <typename T>
    void alltoall(const T *send_data, int send_count, T *recv_data, int recv_count)
    {
        check_type<T>();
        static constexpr auto mpi_type = mpi_type_map<T>::type;
        MPI_Alltoall(send_data, send_count, mpi_type, recv_data, recv_count, mpi_type, m_comm);
    }

    // 只发送一个的情况
    template <typename T>
    void alltoall(const T *send_data, T *recv_data)
    {
        check_type<T>();
        static constexpr auto mpi_type = mpi_type_map<T>::type;
        MPI_Alltoall(send_data, 1, mpi_type, recv_data, 1, mpi_type, m_comm);
    }

  private:
    MPI_Comm m_comm;
};

inline communicator world(MPI_COMM_WORLD);

} // end namespace mpi

#endif // MPI_BASE_HPP