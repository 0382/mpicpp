#pragma once
#ifndef MPI_INFO_HPP
#define MPI_INFO_HPP

#include "error.hpp"
#include <memory>
#include <string>

namespace mpi
{

class info
{
  private:
    MPI_Info m_info;

  public:
    info() : m_info(MPI_INFO_NULL) {}
    info(MPI_Info info) : m_info(info) {}
    static info create()
    {
        MPI_Info t;
        CHECK_MPI(MPI_Info_create(&t));
        return info{t};
    }
    MPI_Info data() const { return m_info; }
    bool is_null() const { return m_info == MPI_INFO_NULL; }
    bool valid() const { return m_info != MPI_INFO_NULL; }
    void free()
    {
        if (is_null())
            return;
        CHECK_MPI(MPI_Info_free(&m_info));
    }
    void set(const char *key, const char *value) { CHECK_MPI(MPI_Info_set(m_info, key, value)); }
    void remove(const char *key) { CHECK_MPI(MPI_Info_delete(m_info, key)); }
    void set(const std::string &key, const std::string &value) { set(key.c_str(), value.c_str()); }
    info copy() const
    {
        MPI_Info t;
        CHECK_MPI(MPI_Info_dup(m_info, &t));
        return info{t};
    }
    std::string get(const char *key) const
    {
        int valuelen;
        int flag;
        CHECK_MPI(MPI_Info_get_valuelen(m_info, key, &valuelen, &flag));
        if (flag)
        {
            auto value = std::make_unique<char[]>(valuelen);
            CHECK_MPI(MPI_Info_get(m_info, key, valuelen, value.get(), &flag));
            return value.get();
        }
        return "";
    }
    std::string get(const std::string &key) const { return get(key.c_str()); }
    std::size_t size() const
    {
        int size;
        CHECK_MPI(MPI_Info_get_nkeys(m_info, &size));
        return size;
    }
    std::string key(std::size_t index) const
    {
        char key[MPI_MAX_INFO_KEY];
        CHECK_MPI(MPI_Info_get_nthkey(m_info, index, key));
        return key;
    }
};

} // end namespace mpi

#endif // MPI_INFO_HPP