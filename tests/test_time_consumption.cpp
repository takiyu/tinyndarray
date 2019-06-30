#define CATCH_CONFIG_MAIN  // Define main()
#include "Catch2/single_include/catch2/catch.hpp"

#define TINYNDARRAY_IMPLEMENTATION
#include "../tinyndarray.h"

#include <chrono>
#include <functional>

using namespace tinyndarray;

constexpr int W = 20000;
constexpr int H = 20000;
constexpr int WH = W * H;
constexpr int N_WORKERS = -1;

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

static Timer g_timer;

static auto MeasureOpTime(std::function<NdArray()> op) {
    g_timer.start();
    auto&& ret = op();
    g_timer.end();
    return std::tuple<float, NdArray>(g_timer.getElapsedMsec(), std::move(ret));
}

template <typename... T>
auto SplitRetItems(T... items) {
    return std::tuple<std::vector<float>, std::vector<NdArray>>(
            {std::get<0>(items)...}, {std::get<1>(items)...});
}

static void PrintTimeResult(const std::string& tag,
                            const std::vector<float>& times,
                            const int time_width) {
    std::cout << "  * " << tag << "  : ";
    std::cout << std::setw(time_width) << times[0] << " ms";
    for (size_t i = 1; i < times.size(); i++) {
        std::cout << ", " << std::setw(time_width) << times[i] << " ms";
    }
    std::cout << std::endl;
}

static void CheckSameNdArray(const NdArray& m1, const NdArray& m2) {
    CHECK(m1.shape() == m2.shape());
    auto&& data1 = m1.data();
    auto&& data2 = m2.data();
    for (int i = 0; i < static_cast<int>(m1.size()); i++) {
        CHECK(data1[i] == Approx(data2[i]));
    }
}

template <typename F, typename... OP>
void TestSingleMultiThread(const std::string& tag, F prep_func, OP... ops) {
    static_assert(0 < sizeof...(ops), "No operation");

    // Print title
    std::cout << "* " << tag << std::endl;

    // Single thread
    NdArray::SetNumWorkers(1);
    const auto& single_rets = SplitRetItems(MeasureOpTime(ops)...);
    const auto& single_times = std::get<0>(single_rets);
    const auto& single_arrays = std::get<1>(single_rets);
    // Print result
    PrintTimeResult("Single", single_times, 5);

    // Prepare for next task
    prep_func();

    // Multi thread
    NdArray::SetNumWorkers(N_WORKERS);
    const auto& multi_rets = SplitRetItems(MeasureOpTime(ops)...);
    const auto& multi_times = std::get<0>(multi_rets);
    const auto& multi_arrays = std::get<1>(multi_rets);
    // Print result
    PrintTimeResult("Multi ", multi_times, 5);

    // Time check
    for (size_t i = 0; i < sizeof...(ops); i++) {
        CHECK(multi_times[i] < single_times[i]);
    }

    // Check array content
    for (size_t i = 0; i < sizeof...(ops); i++) {
        CheckSameNdArray(single_arrays[i], multi_arrays[i]);
    }
}

// -------------------------- Element-wise Operation ---------------------------
TEST_CASE("NdArray Element-wise") {
    SECTION("(NdArray, float)") {
        auto m1 = NdArray::Arange(WH);
        auto m1_move = NdArray::Arange(WH);
        auto m1_move_sub = NdArray::Arange(WH);
        auto m1_cao = NdArray::Arange(WH);
        auto m1_cao_sub = NdArray::Arange(WH);
        TestSingleMultiThread(
                "Element-wise (NdArray, float)",
                [&]() {
                    m1_move = std::move(m1_move_sub);  // Preparation for multi
                    m1_cao = std::move(m1_cao_sub);
                },
                [&]() { return m1 + 1.f; },                  // Basic
                [&]() { return std::move(m1_move) + 1.f; },  // Inplace
                [&]() { return m1_cao += 1.f; });  // Compound Assignment
    }

    SECTION("(NdArray, NdArray) (same-size)") {
        auto m1 = NdArray::Arange(WH);
        auto m1_move = NdArray::Arange(WH);
        auto m1_move_sub = NdArray::Arange(WH);
        auto m1_cao = NdArray::Arange(WH);
        auto m1_cao_sub = NdArray::Arange(WH);
        auto m2 = NdArray::Ones(WH);

        TestSingleMultiThread(
                "Element-wise (NdArray, NdArray) (same-size)",
                [&]() {
                    m1_move = std::move(m1_move_sub);
                    m1_cao = std::move(m1_cao_sub);
                },
                [&]() { return m1 + m2; },                  // Basic
                [&]() { return std::move(m1_move) + m2; },  // Inplace
                [&]() { return m1_cao += m2; });  // Compound Assignment
    }

    SECTION("(NdArray, NdArray) (broadcast) (left-big)") {
        auto m1 = NdArray::Arange(WH).reshape(H, W);
        auto m1_move = NdArray::Arange(WH).reshape(H, W);
        auto m1_move_sub = NdArray::Arange(WH).reshape(H, W);
        auto m1_cao = NdArray::Arange(WH).reshape(H, W);
        auto m1_cao_sub = NdArray::Arange(WH).reshape(H, W);
        auto m2 = NdArray::Ones(W);
        TestSingleMultiThread(
                "Element-wise (NdArray, NdArray) (broadcast) (left-big)",
                [&]() {
                    m1_move = std::move(m1_move_sub);
                    m1_cao = std::move(m1_cao_sub);
                },
                [&]() { return m1 + m2; },                  // Basic
                [&]() { return std::move(m1_move) + m2; },  // Inplace
                [&]() { return m1_cao += m2; });  // Compound Assignment
    }

    SECTION("(NdArray, NdArray) (broadcast) (right-big)") {
        auto m1 = NdArray::Arange(WH).reshape(H, W);
        auto m1_move = NdArray::Arange(WH).reshape(H, W);
        auto m1_move_sub = NdArray::Arange(WH).reshape(H, W);
        auto m2 = NdArray::Ones(W);
        TestSingleMultiThread(
                "Element-wise (NdArray, NdArray) (broadcast) (right-big)",
                [&]() { m1_move = std::move(m1_move_sub); },
                [&]() { return m2 + m1; },                   // Basic
                [&]() { return m2 + std::move(m1_move); });  // Inplace
    }
}

// -------------------------------- Dot product --------------------------------
TEST_CASE("NdArray Dot") {
    SECTION("(1d1d)") {
        auto m1 = NdArray::Ones(16000000);  // 16777216 is limit of float
        auto m2 = NdArray::Ones(16000000);
        TestSingleMultiThread(
                "Dot (1d1d)", [&]() {}, [&]() { return m1.dot(m2); });
    }

    SECTION("(2d2d)") {
        auto m1 = NdArray::Arange(200 * W).reshape(200, W);
        auto m2 = NdArray::Ones(W, 200);
        TestSingleMultiThread(
                "Dot (2d2d)", [&]() {}, [&]() { return m1.dot(m2); });
    }

    SECTION("(NdMd) (left-big)") {
        auto m1 = NdArray::Arange(WH).reshape(H, 1, W);
        auto m2 = NdArray::Ones(W, 1);
        TestSingleMultiThread(
                "Dot (NdMd) (left-big)", [&]() {},
                [&]() { return m1.dot(m2); });
    }

    SECTION("(NdMd) (right-big)") {
        auto m1 = NdArray::Ones(1, H);
        auto m2 = NdArray::Arange(WH).reshape(W, H, 1);
        TestSingleMultiThread(
                "Dot (NdMd) (right-big)", [&]() {},
                [&]() { return m1.dot(m2); });
    }

    /*
    // This type cannot be operated in parallel.
    SECTION("(NdMd) (right-big 2)") {
        auto m1 = NdArray::Ones(1, H);
        auto m2 = NdArray::Arange(WH).reshape(1, H, W);
        TestSingleMultiThread(
                "Dot (NdMd) (right-big 2)", [&]() {},
                [&]() { return m1.dot(m2); });
    }
    */
}

// ------------------------------- Cross product -------------------------------
TEST_CASE("NdArray Cross") {
    SECTION("(NdMd)") {
        auto m1_a = NdArray::Arange(WH * 3).reshape(H, W, 3);
        auto m1_b = NdArray::Arange(WH * 2).reshape(H, W, 2);
        auto m2_a = NdArray::Ones(W, 3);
        auto m2_b = NdArray::Ones(W, 2);
        TestSingleMultiThread(
                "Cross (NdMd)", [&]() {},
                [&]() { return m1_a.cross(m2_a); },   // 3x3
                [&]() { return m1_a.cross(m2_b); },   // 3x2
                [&]() { return m1_b.cross(m2_b); });  // 2x2
    }
}

// ------------------------------- Axis operation ------------------------------
TEST_CASE("NdArray Axis") {
    SECTION("Sum") {
        const int N = 4000;
        auto m1 = NdArray::Ones(N * N).reshape(N, N);  // 16777216 is limit
        TestSingleMultiThread(
                "Sum", [&]() {}, [&]() { return m1.sum(); },
                [&]() { return m1.sum(Axis{1}); });
    }
    SECTION("Max") {
        auto m1 = NdArray::Arange(WH).reshape(W, H);
        TestSingleMultiThread(
                "Max", [&]() {}, [&]() { return m1.max(); },
                [&]() { return m1.max(Axis{1}); });
    }
}
