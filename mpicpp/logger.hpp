#pragma once
#ifndef MPI_LOGGER_HPP
#define MPI_LOGGER_HPP

#include "communicator.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace mpi
{

enum class LogLevel
{
    Error = 0,
    Warning = 1,
    Info = 2,
    Debug = 3,
    Verbose = 4
};

class Logger
{
  public:
    using Logger_clock = std::chrono::system_clock;
    static Logger &instance()
    {
        static Logger log;
        return log;
    }

    void init(LogLevel level)
    {
        m_log_level = level;
        if (world.rank() == 0)
        {
            m_start_time = MPI_Wtime();
        }
        world.broadcast(m_start_time, 0);
    }

    template <typename... Args>
    void error(Args &&...args) const
    {
        write_log(LogLevel::Error, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void error_stop(Args &&...args) const
    {
        write_log(LogLevel::Error, std::forward<Args>(args)...);
        world.abort(-1);
    }

    template <typename... Args>
    void warn(Args &&...args) const
    {
        write_log(LogLevel::Warning, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void info(Args &&...args) const
    {
        write_log(LogLevel::Info, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void debug(Args &&...args) const
    {
        write_log(LogLevel::Debug, std::forward<Args>(args)...);
    }

    template <typename... Args>
    void verbose(Args &&...args) const
    {
        write_log(LogLevel::Verbose, std::forward<Args>(args)...);
    }

  private:
    LogLevel m_log_level;
    double m_start_time;

  private:
    Logger() = default;
    Logger(const Logger &) = delete;
    Logger &operator=(const Logger &) = delete;
    Logger(Logger &&) = delete;
    Logger &operator=(Logger &&) = delete;

    template <typename... Args>
    void write_log(LogLevel level, Args &&...args) const
    {
        using namespace std::chrono_literals;
        static const auto rank_length = std::to_string(world.size()).size();
        if (static_cast<int>(level) > static_cast<int>(m_log_level))
            return;
        auto now_time = MPI_Wtime();
        auto dura = (now_time - m_start_time) * 1s;
        std::ostringstream oss;
        auto hours = std::chrono::floor<std::chrono::hours>(dura).count() % 24;
        auto minutes = std::chrono::floor<std::chrono::minutes>(dura).count() % 60;
        auto seconds = std::chrono::floor<std::chrono::seconds>(dura).count() % 60;
        oss << '[';
        oss << std::setw(rank_length) << world.rank() << '-';
        oss << std::setw(2) << std::setfill('0') << hours << ':';
        oss << std::setw(2) << std::setfill('0') << minutes << ':';
        oss << std::setw(2) << std::setfill('0') << seconds << "] ";
        (oss << ... << args) << '\n';
        std::cout << oss.str();
        std::cout.flush();
    }
};

inline void log_init(LogLevel level) { Logger::instance().init(level); }

template <typename... Args>
inline void log_error(Args &&...args)
{
    Logger::instance().error(std::forward<Args>(args)...);
}

template <typename... Args>
inline void log_error_stop(Args &&...args)
{
    Logger::instance().error_stop(std::forward<Args>(args)...);
}

template <typename... Args>
inline void log_warn(Args &&...args)
{
    Logger::instance().warn(std::forward<Args>(args)...);
}

template <typename... Args>
inline void log_info(Args &&...args)
{
    Logger::instance().info(std::forward<Args>(args)...);
}

template <typename... Args>
inline void log_debug(Args &&...args)
{
    Logger::instance().debug(std::forward<Args>(args)...);
}

template <typename... Args>
inline void log_verbose(Args &&...args)
{
    Logger::instance().verbose(std::forward<Args>(args)...);
}

} // end namespace mpi

#endif // MPI_LOGGER_HPP