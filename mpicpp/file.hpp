#pragma once
#ifndef MPI_FILE_HPP
#define MPI_FILE_HPP

#include "error.hpp"
#include "info.hpp"
#include "status.hpp"
#include "types.hpp"

namespace mpi
{

class file
{
  private:
    MPI_File m_file;

  public:
    file() : m_file(MPI_FILE_NULL) {}
    file(MPI_File file) : m_file(file) {}
    bool is_open() const { return m_file != MPI_FILE_NULL; }
    void close()
    {
        if (!is_open())
            return;
        CHECK_MPI(MPI_File_close(&m_file));
    }
    ~file() { close(); }

    static void remove(const char *filename) { CHECK_MPI(MPI_File_delete(filename, MPI_INFO_NULL)); }
    static void remove(const std::string &filename) { remove(filename.c_str()); }
    static void remove(const char *filename, const info &inf) { CHECK_MPI(MPI_File_delete(filename, inf.data())); }
    static void remove(const std::string &filename, const info &inf) { remove(filename.c_str(), inf); }
    int amode() const
    {
        int amode;
        CHECK_MPI(MPI_File_get_amode(m_file, &amode));
        return amode;
    }
    int atomicity() const
    {
        int flag;
        CHECK_MPI(MPI_File_get_atomicity(m_file, &flag));
        return flag;
    }
    MPI_Offset byte_offset(MPI_Offset offset) const
    {
        MPI_Offset disp;
        CHECK_MPI(MPI_File_get_byte_offset(m_file, offset, &disp));
        return disp;
    }
    mpi::info info() const
    {
        MPI_Info inf;
        CHECK_MPI(MPI_File_get_info(m_file, &inf));
        return mpi::info{inf};
    }
    MPI_Offset position() const
    {
        MPI_Offset offset;
        CHECK_MPI(MPI_File_get_position(m_file, &offset));
        return offset;
    }
    MPI_Offset position_shared() const
    {
        MPI_Offset offset;
        CHECK_MPI(MPI_File_get_position_shared(m_file, &offset));
        return offset;
    }
    MPI_Offset size() const
    {
        MPI_Offset size;
        CHECK_MPI(MPI_File_get_size(m_file, &size));
        return size;
    }

    void seek(MPI_Offset offset, int whence) { CHECK_MPI(MPI_File_seek(m_file, offset, whence)); }
    void seek_shared(MPI_Offset offset, int whence) { CHECK_MPI(MPI_File_seek_shared(m_file, offset, whence)); }
    void set_atomicity(int flag) { CHECK_MPI(MPI_File_set_atomicity(m_file, flag)); }
    void set_info(const mpi::info &inf) { CHECK_MPI(MPI_File_set_info(m_file, inf.data())); }
    void set_size(MPI_Offset size) { CHECK_MPI(MPI_File_set_size(m_file, size)); }

    void sync() { CHECK_MPI(MPI_File_sync(m_file)); }

    template <typename T>
    void read(T *buf, std::size_t count) const
    {
        CHECK_MPI(MPI_File_read(m_file, buf, count, mpi_type<T>(), MPI_STATUS_IGNORE));
    }
    template <typename T>
    void read(T *buf, std::size_t count, status &st) const
    {
        CHECK_MPI(MPI_File_read(m_file, buf, count, mpi_type<T>(), st.ptr()));
    }
    template <typename T>
    void read_all(T *buf, std::size_t count) const
    {
        CHECK_MPI(MPI_File_read_all(m_file, buf, count, mpi_type<T>(), MPI_STATUS_IGNORE));
    }
    template <typename T>
    void read_all(T *buf, std::size_t count, status &st) const
    {
        CHECK_MPI(MPI_File_read_all(m_file, buf, count, mpi_type<T>(), st.ptr()));
    }
    template <typename T>
    void read_at(MPI_Offset offset, T *buf, std::size_t count) const
    {
        CHECK_MPI(MPI_File_read_at(m_file, offset, buf, count, mpi_type<T>(), MPI_STATUS_IGNORE));
    }
    template <typename T>
    void read_at(MPI_Offset offset, T *buf, std::size_t count, status &st) const
    {
        CHECK_MPI(MPI_File_read_at(m_file, offset, buf, count, mpi_type<T>(), st.ptr()));
    }
    template <typename T>
    void read_at_all(MPI_Offset offset, T *buf, std::size_t count) const
    {
        CHECK_MPI(MPI_File_read_at_all(m_file, offset, buf, count, mpi_type<T>(), MPI_STATUS_IGNORE));
    }
    template <typename T>
    void read_at_all(MPI_Offset offset, T *buf, std::size_t count, status &st) const
    {
        CHECK_MPI(MPI_File_read_at_all(m_file, offset, buf, count, mpi_type<T>(), st.ptr()));
    }
    template <typename T>
    void read_at_all_begin(MPI_Offset offset, T *buf, std::size_t count)
    {
        CHECK_MPI(MPI_File_read_at_all_begin(m_file, offset, buf, count, mpi_type<T>()));
    }
    template <typename T>
    void read_at_all_end(T *buf)
    {
        CHECK_MPI(MPI_File_read_at_all_end(m_file, buf, MPI_STATUS_IGNORE));
    }
    template <typename T>
    void read_at_all_end(T *buf, status &st)
    {
        CHECK_MPI(MPI_File_read_at_all_end(m_file, buf, st.ptr()));
    }
    template <typename T>
    void read_ordered(T *buf, std::size_t count)
    {
        CHECK_MPI(MPI_File_read_ordered(m_file, buf, count, mpi_type<T>(), MPI_STATUS_IGNORE));
    }
    template <typename T>
    void read_ordered(T *buf, std::size_t count, status &st)
    {
        CHECK_MPI(MPI_File_read_ordered(m_file, buf, count, mpi_type<T>(), st.ptr()));
    }
    template <typename T>
    void read_ordered_begin(T *buf, std::size_t count)
    {
        CHECK_MPI(MPI_File_read_ordered_begin(m_file, buf, count, mpi_type<T>()));
    }
    template <typename T>
    void read_ordered_end(T *buf)
    {
        CHECK_MPI(MPI_File_read_ordered_end(m_file, buf, MPI_STATUS_IGNORE));
    }
    template <typename T>
    void read_ordered_end(T *buf, status &st)
    {
        CHECK_MPI(MPI_File_read_ordered_end(m_file, buf, st.ptr()));
    }
    template <typename T>
    void read_shared(T *buf, std::size_t count)
    {
        CHECK_MPI(MPI_File_read_shared(m_file, buf, count, mpi_type<T>(), MPI_STATUS_IGNORE));
    }
    template <typename T>
    void read_shared(T *buf, std::size_t count, status &st)
    {
        CHECK_MPI(MPI_File_read_shared(m_file, buf, count, mpi_type<T>(), st.ptr()));
    }

    template <typename T>
    void write(const T *buf, std::size_t count)
    {
        CHECK_MPI(MPI_File_write(m_file, buf, count, mpi_type<T>(), MPI_STATUS_IGNORE));
    }
    template <typename T>
    void write(const T *buf, std::size_t count, status &st)
    {
        CHECK_MPI(MPI_File_write(m_file, buf, count, mpi_type<T>(), st.ptr()));
    }
    template <typename T>
    void write_all(const T *buf, std::size_t count)
    {
        CHECK_MPI(MPI_File_write_all(m_file, buf, count, mpi_type<T>(), MPI_STATUS_IGNORE));
    }
};

} // end namespace mpi

#endif // MPI_FILE_HPP