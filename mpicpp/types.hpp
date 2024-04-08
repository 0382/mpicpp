#pragma once
#ifndef MPI_TYPES_HPP
#define MPI_TYPES_HPP

#include <algorithm>
#include <cassert>
#include <complex>
#include <mpi.h>
#include <type_traits>

namespace mpi
{

// clang-format off
template <typename T> const MPI_Datatype mpi_type() { return MPI_DATATYPE_NULL; }
template <> const MPI_Datatype mpi_type<char>() { return MPI_CHAR; }
template <> const MPI_Datatype mpi_type<unsigned char>() { return MPI_UNSIGNED_CHAR; }
template <> const MPI_Datatype mpi_type<signed char>() { return MPI_SIGNED_CHAR; }
template <> const MPI_Datatype mpi_type<short>() { return MPI_SHORT; }
template <> const MPI_Datatype mpi_type<unsigned short>() { return MPI_UNSIGNED_SHORT; }
template <> const MPI_Datatype mpi_type<int>() { return MPI_INT; }
template <> const MPI_Datatype mpi_type<unsigned>() { return MPI_UNSIGNED; }
template <> const MPI_Datatype mpi_type<long>() { return MPI_LONG; }
template <> const MPI_Datatype mpi_type<unsigned long>() { return MPI_UNSIGNED_LONG; }
template <> const MPI_Datatype mpi_type<long long>() { return MPI_LONG_LONG; }
template <> const MPI_Datatype mpi_type<unsigned long long>() { return MPI_UNSIGNED_LONG_LONG; }
template <> const MPI_Datatype mpi_type<float>() { return MPI_FLOAT; }
template <> const MPI_Datatype mpi_type<double>() { return MPI_DOUBLE; }
template <> const MPI_Datatype mpi_type<long double>() { return MPI_LONG_DOUBLE; }

template <> const MPI_Datatype mpi_type<bool>() { return MPI_CXX_BOOL; }
template <> const MPI_Datatype mpi_type<std::complex<float>>() { return MPI_CXX_FLOAT_COMPLEX; }
template <> const MPI_Datatype mpi_type<std::complex<double>>() { return MPI_CXX_COMPLEX; }
template <> const MPI_Datatype mpi_type<std::complex<long double>>() { return MPI_CXX_LONG_DOUBLE_COMPLEX; }

template <> const MPI_Datatype mpi_type<std::byte>() { return MPI_BYTE; }
// clang-format on

template <typename T>
inline void check_type()
{
    assert(mpi_type<T>() != MPI_DATATYPE_NULL && "Unsupported type");
}

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