#pragma once
#ifndef MPI_TYPES_HPP
#define MPI_TYPES_HPP

#include <mpi.h>

namespace mpi
{

template <typename T>
struct mpi_type_map
{
    static constexpr bool supported = false;
};

template <>
struct mpi_type_map<char>
{
    static constexpr bool supported = true;
    static constexpr MPI_Datatype type = MPI_CHAR;
};

template <>
struct mpi_type_map<unsigned char>
{
    static constexpr bool supported = true;
    static constexpr MPI_Datatype type = MPI_UNSIGNED_CHAR;
};

template <>
struct mpi_type_map<short>
{
    static constexpr bool supported = true;
    static constexpr MPI_Datatype type = MPI_SHORT;
};

template <>
struct mpi_type_map<unsigned short>
{
    static constexpr bool supported = true;
    static constexpr MPI_Datatype type = MPI_UNSIGNED_SHORT;
};

template <>
struct mpi_type_map<int>
{
    static constexpr bool supported = true;
    static constexpr MPI_Datatype type = MPI_INT;
};

template <>
struct mpi_type_map<unsigned int>
{
    static constexpr bool supported = true;
    static constexpr MPI_Datatype type = MPI_UNSIGNED;
};

template <>
struct mpi_type_map<long>
{
    static constexpr bool supported = true;
    static constexpr MPI_Datatype type = MPI_LONG;
};

template <>
struct mpi_type_map<unsigned long>
{
    static constexpr bool supported = true;
    static constexpr MPI_Datatype type = MPI_UNSIGNED_LONG;
};

template <>
struct mpi_type_map<long long>
{
    static constexpr bool supported = true;
    static constexpr MPI_Datatype type = MPI_LONG_LONG;
};

template <>
struct mpi_type_map<unsigned long long>
{
    static constexpr bool supported = true;
    static constexpr MPI_Datatype type = MPI_UNSIGNED_LONG_LONG;
};

template <>
struct mpi_type_map<float>
{
    static constexpr bool supported = true;
    static constexpr MPI_Datatype type = MPI_FLOAT;
};

template <>
struct mpi_type_map<double>
{
    static constexpr bool supported = true;
    static constexpr MPI_Datatype type = MPI_DOUBLE;
};

template <>
struct mpi_type_map<long double>
{
    static constexpr bool supported = true;
    static constexpr MPI_Datatype type = MPI_LONG_DOUBLE;
};

} // end namespace mpi

#endif // MPI_TYPES_HPP