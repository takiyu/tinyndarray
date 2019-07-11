#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

#include "timer.h"  // g_timer

#define TINYNDARRAY_IMPLEMENTATION
#define TINYNDARRAY_NO_INCLUDE

// -----------------------------------------------------------------------------
// Automatic
namespace tnd_auto {
#undef TINYNDARRAY_H_ONCE
#include "../tinyndarray.h"
}  // namespace tnd_auto

// -----------------------------------------------------------------------------
// Col-major
namespace tnd_col {
#undef TINYNDARRAY_H_ONCE
#define TINYNDARRAY_FORCE_DOT_COLMAJOR
#include "../tinyndarray.h"
#undef TINYNDARRAY_FORCE_DOT_COLMAJOR
}  // namespace tnd_col

// -----------------------------------------------------------------------------
// Row major
namespace tnd_row {
#undef TINYNDARRAY_H_ONCE
#define TINYNDARRAY_FORCE_DOT_RAWMAJOR
#include "../tinyndarray.h"
#undef TINYNDARRAY_FORCE_DOT_RAWMAJOR
}  // namespace tnd_row
// -----------------------------------------------------------------------------

using NdArrayAuto = tnd_auto::tinyndarray::NdArray;
using NdArrayCol = tnd_col::tinyndarray::NdArray;
using NdArrayRow = tnd_row::tinyndarray::NdArray;
using namespace tnd_auto::tinyndarray;

// -----------------------------------------------------------------------------

template <typename T>
float TestDotTimeOne(const Shape& l_shape, const Shape& r_shape) {
    try {
        float elapsed = 0.f;
        float count = 0.f;
        while (elapsed < 500.f) {
            // Run
            g_timer.start();
            T::Ones(l_shape).dot(T::Ones(r_shape));
            g_timer.end();
            elapsed += g_timer.getElapsedMsec();
            count += 1.f;
        }
        return elapsed / count;
    } catch (...) {
        return std::numeric_limits<float>::max();
    }
}

void TestDotTime(const Shape& l_shape, const Shape& r_shape) {
    // Print header
    std::stringstream ss;
    ss << l_shape << " @ " << r_shape << ": ";
    const std::string& head_str = ss.str();
    std::cout << head_str;
    const int lack = 40 - static_cast<int>(head_str.size());
    for (int i = 0; i < lack; i++) {
        std::cout << " ";
    }

    // Measure
    const float time_auto = TestDotTimeOne<NdArrayAuto>(l_shape, r_shape);
    std::cout << "a:" << time_auto << "ms" << ",  ";
    const float time_col = TestDotTimeOne<NdArrayCol>(l_shape, r_shape);
    std::cout << "c:" << time_col << "ms" << ",  ";
    const float time_row = TestDotTimeOne<NdArrayRow>(l_shape, r_shape);
    std::cout << "r:" << time_row << "ms" << ",  ";

    // Analyze
    const bool should_col = (time_col < time_row);
    const bool used_col = (std::abs(time_col - time_auto) <
                           std::abs(time_row - time_auto));
    if (used_col) {
        std::cout << "(now: col) ";
    } else {
        std::cout << "(now: row) ";
    }
    if (should_col) {
        std::cout << "(should be col) ";
    } else {
        std::cout << "(should be row) ";
    }
    if (used_col == should_col) {
        // std::cout << "-> OK";
    } else {
        std::cout << "-> NG";
    }
    std::cout << std::endl;
}


int main(int argc, char const* argv[]) {
    (void)argc;
    (void)argv;

    constexpr int W = 20000;
    constexpr int W2 = 2000;
    // constexpr int S = 200;
    constexpr int S = 1;

    TestDotTime({1, W}, {W, 100});
    TestDotTime({10, W}, {W, 100});
    TestDotTime({100, W}, {W, 100});
    TestDotTime({1000, W}, {W, 100});

    TestDotTime({1, W}, {W, 1});
    TestDotTime({10, W}, {W, 10});
    TestDotTime({100, W}, {W, 100});
    TestDotTime({1000, W}, {W, 1000});

    TestDotTime({1, W}, {W, 1000});
    TestDotTime({10, W}, {W, 100});
    TestDotTime({100, W}, {W, 10});
    TestDotTime({1000, W}, {W, 1});

    TestDotTime({100, W}, {W, 1});
    TestDotTime({100, W}, {W, 2});
    TestDotTime({100, W}, {W, 5});
    TestDotTime({100, W}, {W, 7});
    TestDotTime({100, W}, {W, 10});
    TestDotTime({100, W}, {W, 12});
    TestDotTime({100, W}, {W, 15});
    TestDotTime({100, W}, {W, 17});
    TestDotTime({100, W}, {W, 20});
    TestDotTime({100, W}, {W, 50});
    TestDotTime({100, W}, {W, 70});
    TestDotTime({100, W}, {W, 20});
    TestDotTime({100, W}, {W, 50});
    TestDotTime({100, W}, {W, 70});
    TestDotTime({100, W}, {W, 100});
    TestDotTime({100, W}, {W, 1000});

    TestDotTime({W, 1}, {1, W});
    TestDotTime({W, 10}, {10, W});
    TestDotTime({W, 100}, {100, W});

    TestDotTime({10 * W2, 1}, {1, W2});
    TestDotTime({10 * W2, 10}, {10, W2});
    TestDotTime({10 * W2, 100}, {100, W2});

    std::cout << "------------------------------------------" << std::endl;

    TestDotTime({10 * W2, 1}, {1, 100});
    TestDotTime({10 * W2, 10}, {10, 100});
    TestDotTime({10 * W2, 100}, {100, 100});

    TestDotTime({10 * 100, 1}, {1, W2});
    TestDotTime({10 * 100, 10}, {10, W2});
    TestDotTime({10 * 100, 100}, {100, W2});

    TestDotTime({100, W2}, {W2, 20000});
    TestDotTime({20000, W2}, {W2, 100});

    std::cout << "------------------------------------------" << std::endl;

    TestDotTime({1, W}, {S, W, 100});
    TestDotTime({10, W}, {S, W, 100});
    TestDotTime({100, W}, {S, W, 100});
    TestDotTime({1000, W}, {S, W, 100});

    TestDotTime({1, W}, {S, W, 1});
    TestDotTime({10, W}, {S, W, 10});
    TestDotTime({100, W}, {S, W, 100});

    TestDotTime({1, W}, {S, W, 1000});
    TestDotTime({10, W}, {S, W, 100});
    TestDotTime({100, W}, {S, W, 10});
    TestDotTime({1000, W}, {S, W, 1});

    TestDotTime({100, W}, {S, W, 1});
    TestDotTime({100, W}, {S, W, 10});
    TestDotTime({100, W}, {S, W, 100});

    return 0;
}
