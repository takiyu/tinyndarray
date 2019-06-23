#define CATCH_CONFIG_MAIN  // Define main()
#include "Catch2/single_include/catch2/catch.hpp"

#define TINYNDARRAY_IMPLEMENTATION
#include "../tinyndarray.h"

#include <chrono>

using namespace tinyndarray;

constexpr int W = 20000;
constexpr int H = 20000;
constexpr int WH = W * H;

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
        return std::chrono::duration_cast<std::chrono::milliseconds>(m_end -
                                                                     m_start)
                .count();
    }

private:
    std::chrono::system_clock::time_point m_start, m_end;
};

Timer g_timer;

TEST_CASE("NdArray") {
    SECTION("Simple op (1 args)") {
        auto m1 = NdArray::Arange(WH);
        auto m1_move1 = NdArray::Arange(WH);
        auto m1_move2 = NdArray::Arange(WH);
        auto m1_cao1 = NdArray::Arange(WH);
        auto m1_cao2 = NdArray::Arange(WH);

        // Single thread
        N_WORKERS = 1;
        // Basic
        g_timer.start();
        auto ret1_single = m1 + 1.f;
        g_timer.end();
        const float t1_single = g_timer.getElapsedMsec();
        // In-place
        g_timer.start();
        auto ret2_single = std::move(m1_move1) + 1.f;
        g_timer.end();
        const float t2_single = g_timer.getElapsedMsec();
        // Compound Assignment
        g_timer.start();
        m1_cao1 += 1.f;
        g_timer.end();
        const float t3_single = g_timer.getElapsedMsec();

        // Multi thread
        N_WORKERS = -1;
        // Basic
        g_timer.start();
        auto ret1_multi = m1 + 1.f;
        g_timer.end();
        const float t1_multi = g_timer.getElapsedMsec();
        // In-place
        g_timer.start();
        auto ret2_multi = std::move(m1_move2) + 1.f;
        g_timer.end();
        const float t2_multi = g_timer.getElapsedMsec();
        // Compound Assignment
        g_timer.start();
        m1_cao2 += 1.f;
        g_timer.end();
        const float t3_multi = g_timer.getElapsedMsec();

        std::cout << "* Simple op (1 args)" << std::endl;
        std::cout << "  * Single : " << t1_single << " ms" << std::endl;
        std::cout << "             " << t2_single << " ms" << std::endl;
        std::cout << "             " << t3_single << " ms" << std::endl;
        std::cout << "  * Multi :  " << t1_multi << " ms" << std::endl;
        std::cout << "             " << t2_multi << " ms" << std::endl;
        std::cout << "             " << t3_multi << " ms" << std::endl;

        REQUIRE(t1_multi < t1_single);
        REQUIRE(t2_multi < t2_single);
        REQUIRE(t3_multi < t3_single);
    }

    SECTION("Simple op (2 args)") {
        auto m1 = NdArray::Arange(WH);
        auto m1_move1 = NdArray::Arange(WH);
        auto m1_move2 = NdArray::Arange(WH);
        auto m1_cao1 = NdArray::Arange(WH);
        auto m1_cao2 = NdArray::Arange(WH);
        auto m2 = NdArray::Zeros(WH);

        // Single thread
        N_WORKERS = 1;
        // Basic
        g_timer.start();
        auto ret1_single = m1 + m2;
        g_timer.end();
        const float t1_single = g_timer.getElapsedMsec();
        // In-place
        g_timer.start();
        auto ret2_single = std::move(m1_move1) + m2;
        g_timer.end();
        const float t2_single = g_timer.getElapsedMsec();
        // Compound Assignment
        g_timer.start();
        m1_cao1 += m2;
        g_timer.end();
        const float t3_single = g_timer.getElapsedMsec();

        // Multi thread
        N_WORKERS = -1;
        // Basic
        g_timer.start();
        auto ret1_multi = m1 + m2;
        g_timer.end();
        const float t1_multi = g_timer.getElapsedMsec();
        // In-place
        g_timer.start();
        auto ret2_multi = std::move(m1_move2) + m2;
        g_timer.end();
        const float t2_multi = g_timer.getElapsedMsec();
        // Compound Assignment
        g_timer.start();
        m1_cao2 += m2;
        g_timer.end();
        const float t3_multi = g_timer.getElapsedMsec();

        std::cout << "* Simple op (2 args)" << std::endl;
        std::cout << "  * Single : " << t1_single << " ms" << std::endl;
        std::cout << "             " << t2_single << " ms" << std::endl;
        std::cout << "             " << t3_single << " ms" << std::endl;
        std::cout << "  * Multi :  " << t1_multi << " ms" << std::endl;
        std::cout << "             " << t2_multi << " ms" << std::endl;
        std::cout << "             " << t3_multi << " ms" << std::endl;

        REQUIRE(t1_multi < t1_single);
        REQUIRE(t2_multi < t2_single);
        REQUIRE(t3_multi < t3_single);
    }

    SECTION("Broadcast op") {
        auto m1 = NdArray::Arange(WH).reshape(H, W);
        auto m1_move1 = NdArray::Arange(WH).reshape(H, W);
        auto m1_move2 = NdArray::Arange(WH).reshape(H, W);
        auto m1_cao1 = NdArray::Arange(WH).reshape(H, W);
        auto m1_cao2 = NdArray::Arange(WH).reshape(H, W);
        auto m2 = NdArray::Zeros(W);

        // Single thread
        N_WORKERS = 1;
        // Basic
        g_timer.start();
        auto ret1_single = m1 + m2;
        g_timer.end();
        const float t1_single = g_timer.getElapsedMsec();
        // In-place
        g_timer.start();
        auto ret2_single = std::move(m1_move1) + m2;
        g_timer.end();
        const float t2_single = g_timer.getElapsedMsec();
        // Compound Assignment
        g_timer.start();
        m1_cao1 += m2;
        g_timer.end();
        const float t3_single = g_timer.getElapsedMsec();

        // Multi thread
        N_WORKERS = -1;
        // Basic
        g_timer.start();
        auto ret1_multi = m1 + m2;
        g_timer.end();
        const float t1_multi = g_timer.getElapsedMsec();
        // In-place
        g_timer.start();
        auto ret2_multi = std::move(m1_move2) + m2;
        g_timer.end();
        const float t2_multi = g_timer.getElapsedMsec();
        // Compound Assignment
        g_timer.start();
        m1_cao2 += m2;
        g_timer.end();
        const float t3_multi = g_timer.getElapsedMsec();

        std::cout << "* Broadcast op" << std::endl;
        std::cout << "  * Single : " << t1_single << " ms" << std::endl;
        std::cout << "             " << t2_single << " ms" << std::endl;
        std::cout << "             " << t3_single << " ms" << std::endl;
        std::cout << "  * Multi :  " << t1_multi << " ms" << std::endl;
        std::cout << "             " << t2_multi << " ms" << std::endl;
        std::cout << "             " << t3_multi << " ms" << std::endl;

        //         REQUIRE(t1_multi < t1_single);
        //         REQUIRE(t2_multi < t2_single);
        //         REQUIRE(t3_multi < t3_single);
    }
}
