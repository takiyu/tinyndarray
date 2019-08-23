#ifndef TINYNDARRAY_TIMER_H
#define TINYNDARRAY_TIMER_H

#include <chrono>

class Timer {
public:
    Timer() {}

    void start() {
        m_start = std::chrono::system_clock::now();
    }

    void end() {
        m_end = std::chrono::system_clock::now();
    }

    float getElapsedMsec() const {
        using namespace std::chrono;
        return static_cast<float>(
                duration_cast<milliseconds>(m_end - m_start).count());
    }

private:
    std::chrono::system_clock::time_point m_start, m_end;
};

static Timer g_timer;

#endif /* end of include guard */
