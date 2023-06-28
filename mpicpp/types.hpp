#pragma once
#ifndef MPI_TYPES_HPP
#define MPI_TYPES_HPP

#include <algorithm>
#include <openmpi/mpi.h>
#include <type_traits>

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

// TODO: not good, maybe a better implementation.
struct op
{
    static MPI_Op null() { return MPI_OP_NULL; }
    static MPI_Op max() { return MPI_MAX; }
    static MPI_Op min() { return MPI_MIN; }
    static MPI_Op sum() { return MPI_SUM; }
    static MPI_Op prod() { return MPI_PROD; }
    static MPI_Op land() { return MPI_LAND; }
    static MPI_Op lor() { return MPI_LOR; }
    static MPI_Op lxor() { return MPI_LXOR; }
    static MPI_Op band() { return MPI_BAND; }
    static MPI_Op bor() { return MPI_BOR; }
    static MPI_Op bxor() { return MPI_BXOR; }
    static MPI_Op minloc() { return MPI_MINLOC; }
    static MPI_Op maxloc() { return MPI_MAXLOC; }
    static MPI_Op replace() { return MPI_REPLACE; }

    template <typename T, typename Func>
    static MPI_Op custom(bool commute)
    {
        static_assert(std::is_invocable<Func, T, T>::value, "Func must be T(T,T)");
        MPI_Op op;
        MPI_Op_create(&func<T, Func>, commute, &op);
        return op;
    }

  private:
    template <typename T, typename Func>
    static void func(void *invec, void *inoutvec, int *len, MPI_Datatype *datatype)
    {
        T *in = static_cast<T *>(invec);
        T *out = static_cast<T *>(inoutvec);
        Func func;
        std::transform(in, in + *len, out, out, func);
    }
};

} // end namespace mpi

#endif // MPI_TYPES_HPP