#include "Catch2/single_include/catch2/catch.hpp"

#include "../tinyndarray.h"

#include <iomanip>

using namespace tinyndarray;

static void CheckNdArray(const NdArray& m, const std::string& str,
                         int precision = -1) {
    std::stringstream ss;
    if (0 < precision) {
        ss << std::setprecision(4);
    }
    ss << m;
    CHECK(ss.str() == str);
}

static void CheckNdArrayInplace(NdArray&& x, const std::string& str,
                                std::function<NdArray(NdArray&&)> f,
                                int precision = -1) {
    uintptr_t x_id = x.id();
    const NdArray& y = f(std::move(x));
    CHECK(y.id() == x_id);
    CheckNdArray(y, str, precision);
}

static void CheckNdArrayInplace(NdArray&& lhs, NdArray&& rhs,
                                const std::string& str,
                                std::function<NdArray(NdArray&&, NdArray&&)> f,
                                int precision = -1) {
    uintptr_t l_id = lhs.id();
    uintptr_t r_id = rhs.id();
    const NdArray& ret = f(std::move(lhs), std::move(rhs));
    CHECK((ret.id() == l_id || ret.id() == r_id));
    CheckNdArray(ret, str, precision);
}

static void CheckNdArrayInplace(
        const NdArray& lhs, NdArray&& rhs, const std::string& str,
        std::function<NdArray(const NdArray&, NdArray&&)> f,
        int precision = -1) {
    uintptr_t l_id = lhs.id();
    uintptr_t r_id = rhs.id();
    const NdArray& ret = f(lhs, std::move(rhs));
    CHECK((ret.id() == l_id || ret.id() == r_id));
    CheckNdArray(ret, str, precision);
}

static void CheckNdArrayInplace(
        NdArray&& lhs, const NdArray& rhs, const std::string& str,
        std::function<NdArray(NdArray&&, const NdArray&)> f,
        int precision = -1) {
    uintptr_t l_id = lhs.id();
    uintptr_t r_id = rhs.id();
    const NdArray& ret = f(std::move(lhs), rhs);
    CHECK((ret.id() == l_id || ret.id() == r_id));
    CheckNdArray(ret, str, precision);
}

static void CheckNdArrayInplace(NdArray&& lhs, float rhs,
                                const std::string& str,
                                std::function<NdArray(NdArray&&, float)> f,
                                int precision = -1) {
    uintptr_t l_id = lhs.id();
    const NdArray& ret = f(std::move(lhs), rhs);
    CHECK((ret.id() == l_id));
    CheckNdArray(ret, str, precision);
}

static void CheckNdArrayInplace(float lhs, NdArray&& rhs,
                                const std::string& str,
                                std::function<NdArray(float, NdArray&&)> f,
                                int precision = -1) {
    uintptr_t r_id = rhs.id();
    const NdArray& ret = f(lhs, std::move(rhs));
    CHECK((ret.id() == r_id));
    CheckNdArray(ret, str, precision);
}

static void CheckNdArrayNotInplace(NdArray&& x, const std::string& str,
                                   std::function<NdArray(const NdArray&)> f,
                                   int precision = -1) {
    uintptr_t x_id = x.id();
    const NdArray& y = f(x);
    CHECK(y.id() != x_id);
    CheckNdArray(y, str, precision);
}

static bool IsSameNdArray(const NdArray& m1, const NdArray& m2) {
    if (m1.shape() != m2.shape()) {
        return false;
    }
    auto&& data1 = m1.data();
    auto&& data2 = m2.data();
    for (int i = 0; i < static_cast<int>(m1.size()); i++) {
        if (data1[i] != data2[i]) {
            return false;
        }
    }
    return true;
}

static void ResolveAmbiguous(NdArray& x) {
    for (auto&& v : x) {
        if (std::isnan(v)) {
            v = std::abs(v);
        }
        if (v == -0.f) {
            v = 0.f;
        }
    }
}

TEST_CASE("NdArray") {
    // -------------------------- Basic construction ---------------------------
    SECTION("Empty") {
        const NdArray m1;
        CHECK(m1.empty());
        CHECK(m1.size() == 0);
        CHECK(m1.shape() == Shape{0});
        CHECK(m1.ndim() == 1);
    }

    // --------------------------- Float initializer ---------------------------
    SECTION("Float initializer") {
        const NdArray m1 = {1.f, 2.f, 3.f};
        const NdArray m2 = {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
        const NdArray m3 = {{{1.f, 2.f}}, {{3.f, 4.f}}, {{2.f, 3.f}}};
        CHECK(m1.shape() == Shape{3});
        CHECK(m2.shape() == Shape{2, 3});
        CHECK(m3.shape() == Shape{3, 1, 2});
        CheckNdArray(m1, "[1, 2, 3]");
        CheckNdArray(m2,
                     "[[1, 2, 3],\n"
                     " [4, 5, 6]]");
        CheckNdArray(m3,
                     "[[[1, 2]],\n"
                     " [[3, 4]],\n"
                     " [[2, 3]]]");
    }

    SECTION("Float initializer invalid") {
        CHECK_NOTHROW(NdArray{{{1.f, 2.f}}, {{3.f, 4.f}}, {{1.f, 2.f}}});
        CHECK_THROWS(NdArray{{{1, 2}}, {}});
        CHECK_THROWS(NdArray{{{1.f, 2.f}}, {{3.f, 4.f}}, {{1.f, 2.f, 3.f}}});
    }

    SECTION("Confusable initializers") {
        const NdArray m1 = {1.f, 2.f, 3.f};  // Float initializer
        const NdArray m2 = {1, 2, 3};        // Shape (int) initalizer
        const NdArray m3 = {{1, 2, 3}};      // Float initializer due to nest
        CHECK(m1.shape() == Shape{3});
        CHECK(m2.shape() == Shape{1, 2, 3});
        CHECK(m3.shape() == Shape{1, 3});
    }

    // --------------------------- Static initializer --------------------------
    SECTION("Empty/Ones/Zeros") {
        const NdArray m1({2, 5});  // Same as Empty
        const auto m2 = NdArray::Empty({2, 5});
        const auto m3 = NdArray::Zeros({2, 5});
        const auto m4 = NdArray::Ones({2, 5});
        CHECK(m1.shape() == Shape{2, 5});
        CHECK(m2.shape() == Shape{2, 5});
        CHECK(m3.shape() == Shape{2, 5});
        CHECK(m4.shape() == Shape{2, 5});
        CheckNdArray(m3,
                     "[[0, 0, 0, 0, 0],\n"
                     " [0, 0, 0, 0, 0]]");
        CheckNdArray(m4,
                     "[[1, 1, 1, 1, 1],\n"
                     " [1, 1, 1, 1, 1]]");
    }

    SECTION("Empty/Ones/Zeros by template") {
        const NdArray m1({2, 5});  // No template support
        const auto m2 = NdArray::Empty(2, 5);
        const auto m3 = NdArray::Zeros(2, 5);
        const auto m4 = NdArray::Ones(2, 5);
        CHECK(m1.shape() == Shape{2, 5});
        CHECK(m2.shape() == Shape{2, 5});
        CHECK(m3.shape() == Shape{2, 5});
        CHECK(m4.shape() == Shape{2, 5});
        CheckNdArray(m3,
                     "[[0, 0, 0, 0, 0],\n"
                     " [0, 0, 0, 0, 0]]");
        CheckNdArray(m4,
                     "[[1, 1, 1, 1, 1],\n"
                     " [1, 1, 1, 1, 1]]");
    }

    SECTION("Arange") {
        const auto m1 = NdArray::Arange(10.f);
        const auto m2 = NdArray::Arange(0.f, 10.f, 1.f);
        const auto m3 = NdArray::Arange(5.f, 5.5f, 0.1f);
        CHECK(m1.shape() == Shape{10});
        CHECK(m2.shape() == Shape{10});
        CHECK(m3.shape() == Shape{5});
        CheckNdArray(m1, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
        CheckNdArray(m2, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
        CheckNdArray(m3, "[5, 5.1, 5.2, 5.3, 5.4]");
    }

    // --------------------------------- Random --------------------------------
    SECTION("Random uniform") {
        NdArray::Seed(0);
        const auto m1 = NdArray::Uniform({2, 3});
        NdArray::Seed(0);
        const auto m2 = NdArray::Uniform({2, 3});
        NdArray::Seed(1);
        const auto m3 = NdArray::Uniform({2, 3});
        CHECK(IsSameNdArray(m1, m2));
        CHECK(!IsSameNdArray(m1, m3));
    }

    SECTION("Random normal") {
        NdArray::Seed(0);
        const auto m1 = NdArray::Normal({2, 3});
        NdArray::Seed(0);
        const auto m2 = NdArray::Normal({2, 3});
        NdArray::Seed(1);
        const auto m3 = NdArray::Normal({2, 3});
        CHECK(IsSameNdArray(m1, m2));
        CHECK(!IsSameNdArray(m1, m3));
    }

    // ------------------------------ Basic method -----------------------------
    SECTION("Basic method") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = m1.copy();
        CheckNdArray(m2,
                     "[[0, 1, 2],\n"
                     " [3, 4, 5]]");
        m2.fill(-1);
        CheckNdArray(m1,
                     "[[0, 1, 2],\n"
                     " [3, 4, 5]]");
        CheckNdArray(m2,
                     "[[-1, -1, -1],\n"
                     " [-1, -1, -1]]");
        CHECK(m1.ndim() == 2);
        CHECK(m1.flatten().ndim() == 1);
        CHECK(m1.flatten().id() != m1.id());  // Copy
        CHECK(m1.ravel().ndim() == 1);
        CHECK(m1.ravel().id() == m1.id());  // Same instance

        m1.resize({2, 2});
        CheckNdArray(m1,
                     "[[0, 1],\n"
                     " [2, 3]]");
        m1.resize({2, 4});
        CheckNdArray(m1,
                     "[[0, 1, 2, 3],\n"
                     " [0, 0, 0, 0]]");
    }

    // ------------------------------- Begin/End -------------------------------
    SECTION("Begin/End") {
        auto m1 = NdArray::Arange(1.f, 10.01f);
        // C++11 for-loop
        float sum1 = 0.f;
        for (auto&& v : m1) {
            sum1 += v;
        }
        CHECK(sum1 == Approx(55.f));
        // std library
        float sum2 = std::accumulate(m1.begin(), m1.end(), 0.f);
        CHECK(sum2 == Approx(55.f));
    }

    // ------------------------------- Float cast ------------------------------
    SECTION("Float cast") {
        auto m1 = NdArray::Ones({1, 1});
        auto m2 = NdArray::Ones({1, 2});
        CHECK(static_cast<float>(m1) == 1);
        CHECK_THROWS(static_cast<float>(m2));
    }

    // ------------------------------ Index access -----------------------------
    SECTION("Index access by []") {
        auto m1 = NdArray::Arange(12.f);
        auto m2 = NdArray::Arange(12.f).reshape({3, 4});
        auto m3 = NdArray::Arange(12.f).reshape({2, 2, -1});
        m1[3] = -1.f;
        m1[-2] = -2.f;
        m2[{1, 1}] = -1.f;
        m2[{-1, 3}] = -2.f;
        m3[{1, 1, 2}] = -1.f;
        m3[{0, 1, -2}] = -2.f;
        CheckNdArray(m1, "[0, 1, 2, -1, 4, 5, 6, 7, 8, 9, -2, 11]");
        CheckNdArray(m2,
                     "[[0, 1, 2, 3],\n"
                     " [4, -1, 6, 7],\n"
                     " [8, 9, 10, -2]]");
        CheckNdArray(m3,
                     "[[[0, 1, 2],\n"
                     "  [3, -2, 5]],\n"
                     " [[6, 7, 8],\n"
                     "  [9, 10, -1]]]");
    }

    SECTION("Index access by ()") {
        auto m1 = NdArray::Arange(12.f);
        auto m2 = NdArray::Arange(12.f).reshape({3, 4});
        auto m3 = NdArray::Arange(12.f).reshape({2, 2, -1});
        m1(3) = -1.f;
        m1(-2) = -2.f;
        m2(1, 1) = -1.f;
        m2(-1, 3) = -2.f;
        m3(1, 1, 2) = -1.f;
        m3(0, 1, -2) = -2.f;
        CheckNdArray(m1, "[0, 1, 2, -1, 4, 5, 6, 7, 8, 9, -2, 11]");
        CheckNdArray(m2,
                     "[[0, 1, 2, 3],\n"
                     " [4, -1, 6, 7],\n"
                     " [8, 9, 10, -2]]");
        CheckNdArray(m3,
                     "[[[0, 1, 2],\n"
                     "  [3, -2, 5]],\n"
                     " [[6, 7, 8],\n"
                     "  [9, 10, -1]]]");
    }

    SECTION("Index access by [] (const)") {
        const auto m1 = NdArray::Arange(12.f).reshape({1, 4, 3});
        CHECK(m1[{0, 2, 1}] == 7);
        CHECK(m1[{0, -1, 1}] == 10);
    }

    SECTION("Index access by () (const)") {
        const auto m1 = NdArray::Arange(12.f).reshape({1, 4, 3});
        CHECK(m1(0, 2, 1) == 7);
        CHECK(m1(0, -1, 1) == 10);
    }

    // -------------------------------- Reshape --------------------------------
    SECTION("Reshape") {
        auto m1 = NdArray::Arange(12.f);
        auto m2 = m1.reshape({3, 4});
        auto m3 = m2.reshape({2, -1});
        auto m4 = m3.reshape({2, 2, -1});
        CheckNdArray(m1, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]");
        CheckNdArray(m2,
                     "[[0, 1, 2, 3],\n"
                     " [4, 5, 6, 7],\n"
                     " [8, 9, 10, 11]]");
        CheckNdArray(m3,
                     "[[0, 1, 2, 3, 4, 5],\n"
                     " [6, 7, 8, 9, 10, 11]]");
        CheckNdArray(m4,
                     "[[[0, 1, 2],\n"
                     "  [3, 4, 5]],\n"
                     " [[6, 7, 8],\n"
                     "  [9, 10, 11]]]");
    }

    SECTION("Reshape by template") {
        auto m1 = NdArray::Arange(12.f);
        auto m2 = m1.reshape(3, 4);
        auto m3 = m2.reshape(2, -1);
        auto m4 = m3.reshape(2, 2, -1);
        CheckNdArray(m1, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]");
        CheckNdArray(m2,
                     "[[0, 1, 2, 3],\n"
                     " [4, 5, 6, 7],\n"
                     " [8, 9, 10, 11]]");
        CheckNdArray(m3,
                     "[[0, 1, 2, 3, 4, 5],\n"
                     " [6, 7, 8, 9, 10, 11]]");
        CheckNdArray(m4,
                     "[[[0, 1, 2],\n"
                     "  [3, 4, 5]],\n"
                     " [[6, 7, 8],\n"
                     "  [9, 10, 11]]]");
    }

    SECTION("Reshape with value change") {
        auto m1 = NdArray::Arange(12.f);
        auto m2 = m1.reshape({3, 4});
        auto m3 = m2.reshape({2, -1});
        auto m4 = m3.reshape({2, 2, -1});
        m1.data()[0] = -1.f;
        CheckNdArray(m1, "[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]");
        CheckNdArray(m2,
                     "[[-1, 1, 2, 3],\n"
                     " [4, 5, 6, 7],\n"
                     " [8, 9, 10, 11]]");
        CheckNdArray(m3,
                     "[[-1, 1, 2, 3, 4, 5],\n"
                     " [6, 7, 8, 9, 10, 11]]");
        CheckNdArray(m4,
                     "[[[-1, 1, 2],\n"
                     "  [3, 4, 5]],\n"
                     " [[6, 7, 8],\n"
                     "  [9, 10, 11]]]");
    }

    SECTION("Reshape invalid") {
        auto m1 = NdArray::Arange(12.f);
        CHECK_THROWS(m1.reshape({5, 2}));
        CHECK_THROWS(m1.reshape({-1, -1}));
    }

    SECTION("Flatten") {
        auto m1 = NdArray::Arange(12.f).reshape(2, 2, 3);
        auto m2 = m1.flatten();
        CheckNdArray(m2, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]");
    }

    // --------------------------------- Slice ---------------------------------
    SECTION("Slice 2-dim") {
        auto m1 = NdArray::Arange(16.f).reshape(4, 4);
        auto m2 = m1.slice({{1, 3}, {1, 3}});
        auto m3 = m1.slice({1, 3}, {0, 4});
        auto m4 = m1.slice({0, 4}, {1, 3});
        auto m5 = m1.slice({1, -1}, {0, 100000});
        CHECK(m1.shape() == Shape{4, 4});
        CHECK(m2.shape() == Shape{2, 2});
        CHECK(m3.shape() == Shape{2, 4});
        CHECK(m4.shape() == Shape{4, 2});
        CHECK(m5.shape() == Shape{2, 4});
        CheckNdArray(m1,
                     "[[0, 1, 2, 3],\n"
                     " [4, 5, 6, 7],\n"
                     " [8, 9, 10, 11],\n"
                     " [12, 13, 14, 15]]");
        CheckNdArray(m2,
                     "[[5, 6],\n"
                     " [9, 10]]");
        CheckNdArray(m3,
                     "[[4, 5, 6, 7],\n"
                     " [8, 9, 10, 11]]");
        CheckNdArray(m4,
                     "[[1, 2],\n"
                     " [5, 6],\n"
                     " [9, 10],\n"
                     " [13, 14]]");
        CheckNdArray(m5,
                     "[[4, 5, 6, 7],\n"
                     " [8, 9, 10, 11]]");
    }

    SECTION("Slice high-dim") {
        auto m1 = NdArray::Arange(256.f).reshape(4, 4, 4, 4);
        auto m2 = m1.slice({{1, 3}, {1, 3}, {1, 3}, {1, 3}});
        auto m3 = m1.slice({1, 3}, {1, 3}, {1, 3}, {1, 3});
        CHECK(m1.shape() == Shape{4, 4, 4, 4});
        CHECK(m2.shape() == Shape{2, 2, 2, 2});
        CHECK(m3.shape() == Shape{2, 2, 2, 2});
        CheckNdArray(m2,
                     "[[[[85, 86],\n"
                     "   [89, 90]],\n"
                     "  [[101, 102],\n"
                     "   [105, 106]]],\n"
                     " [[[149, 150],\n"
                     "   [153, 154]],\n"
                     "  [[165, 166],\n"
                     "   [169, 170]]]]");
        CheckNdArray(m3,
                     "[[[[85, 86],\n"
                     "   [89, 90]],\n"
                     "  [[101, 102],\n"
                     "   [105, 106]]],\n"
                     " [[[149, 150],\n"
                     "   [153, 154]],\n"
                     "  [[165, 166],\n"
                     "   [169, 170]]]]");
    }

    // ------------------------------ Dot product ------------------------------
    SECTION("Dot (empty)") {
        // Empty array
        auto m1 = NdArray::Arange(0.f);
        CHECK_THROWS(m1.dot(m1));
    }

    SECTION("Dot (scalar)") {
        // Scalar multiply
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m1_a1 = m1.dot(2.f);
        auto m1_a2 = m1.dot(NdArray{2.f});
        auto m1_b = NdArray({2.f}).dot(m1);
        CheckNdArray(m1_a1,
                     "[[0, 2, 4],\n"
                     " [6, 8, 10]]");
        CheckNdArray(m1_a2,
                     "[[0, 2, 4],\n"
                     " [6, 8, 10]]");
        CheckNdArray(m1_b,
                     "[[0, 2, 4],\n"
                     " [6, 8, 10]]");
    }

    SECTION("Dot (1D, 1D)") {
        // Inner product of vectors
        auto m1 = NdArray::Arange(3.f);
        auto m2 = NdArray::Ones(3);
        float m11 = m1.dot(m1);
        float m12 = m1.dot(m2);
        CHECK(m11 == Approx(5.f));
        CHECK(m12 == Approx(3.f));
        // Shape mismatch
        auto m3 = NdArray::Arange(4.f);
        CHECK_THROWS(m1.dot(m3));
    }

    SECTION("Dot (2D, 2D)") {
        // Inner product of 2D matrix
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(6.f).reshape(3, 2);
        auto m12 = m1.dot(m2);
        CheckNdArray(m12,
                     "[[10, 13],\n"
                     " [28, 40]]");
        // Shape mismatch
        CHECK_THROWS(m1.dot(m1));
    }

    SECTION("Dot (2D, 1D)") {
        // Inner product of 2D matrix and vector (2D, 1D)
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(3.f);
        auto m12 = m1.dot(m2);
        CheckNdArray(m12, "[5, 14]");
        // Shape mismatch
        auto m3 = NdArray::Arange(2.f);
        CHECK_THROWS(m1.dot(m3));
    }

    SECTION("Dot (ND, 1D)") {
        // Inner product of ND matrix and vector (ND, 1D)
        auto m1 = NdArray::Arange(12.f).reshape(2, 2, 3);
        auto m2 = NdArray::Arange(3.f);
        auto m12 = m1.dot(m2);
        CheckNdArray(m12,
                     "[[5, 14],\n"
                     " [23, 32]]");
    }

    SECTION("Dot (ND, MD)") {
        // Inner product of ND matrix and MD matrix
        auto m1 = NdArray::Arange(12.f).reshape(2, 3, 2);
        auto m2 = NdArray::Arange(6.f).reshape(2, 3);
        auto m3 = NdArray::Arange(12.f).reshape(3, 2, 2);
        auto m12 = m1.dot(m2);
        auto m13 = m1.dot(m3);
        CHECK(m12.shape() == Shape{2, 3, 3});
        CHECK(m13.shape() == Shape{2, 3, 3, 2});
        CheckNdArray(m12,
                     "[[[3, 4, 5],\n"
                     "  [9, 14, 19],\n"
                     "  [15, 24, 33]],\n"
                     " [[21, 34, 47],\n"
                     "  [27, 44, 61],\n"
                     "  [33, 54, 75]]]");
        CheckNdArray(m13,
                     "[[[[2, 3],\n"
                     "   [6, 7],\n"
                     "   [10, 11]],\n"
                     "  [[6, 11],\n"
                     "   [26, 31],\n"
                     "   [46, 51]],\n"
                     "  [[10, 19],\n"
                     "   [46, 55],\n"
                     "   [82, 91]]],\n"
                     " [[[14, 27],\n"
                     "   [66, 79],\n"
                     "   [118, 131]],\n"
                     "  [[18, 35],\n"
                     "   [86, 103],\n"
                     "   [154, 171]],\n"
                     "  [[22, 43],\n"
                     "   [106, 127],\n"
                     "   [190, 211]]]]");
    }

    // ----------------------------- Matmul product ----------------------------
    SECTION("Matmul (2D, 2D)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(6.f).reshape(3, 2);
        auto m12 = Matmul(m1, m2);
        CheckNdArray(m12,
                     "[[10, 13],\n"
                     " [28, 40]]");
    }

    SECTION("Matmul (2D, 3D)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(6.f).reshape(1, 3, 2);
        auto m12 = Matmul(m1, m2);
        CheckNdArray(m12,
                     "[[[10, 13],\n"
                     "  [28, 40]]]");
    }

    SECTION("Matmul (3D, 2D)") {
        auto m1 = NdArray::Arange(6.f).reshape(1, 2, 3);
        auto m2 = NdArray::Arange(6.f).reshape(3, 2);
        auto m12 = Matmul(m1, m2);
        CheckNdArray(m12,
                     "[[[10, 13],\n"
                     "  [28, 40]]]");
    }

    SECTION("Matmul (1D, 2D)") {
        auto m1 = NdArray::Arange(3.f);
        auto m2 = NdArray::Arange(6.f).reshape(3, 2);
        auto m12 = Matmul(m1, m2);
        CheckNdArray(m12, "[10, 13]");
    }

    SECTION("Matmul (2D, 1D)") {
        auto m1 = NdArray::Arange(6.f).reshape(3, 2);
        auto m2 = NdArray::Arange(2.f);
        auto m12 = Matmul(m1, m2);
        CheckNdArray(m12, "[1, 3, 5]");
    }

    SECTION("Matmul (1D, ND)") {
        auto m1 = NdArray::Arange(3.f);
        auto m2 = NdArray::Arange(12.f).reshape(2, 1, 3, 2);
        auto m12 = Matmul(m1, m2);
        CheckNdArray(m12,
                     "[[[10, 13]],\n"
                     " [[28, 31]]]");
    }

    SECTION("Matmul (2D, ND)") {
        auto m1 = NdArray::Arange(12.f).reshape(2, 1, 3, 2);
        auto m2 = NdArray::Arange(2.f);
        auto m12 = Matmul(m1, m2);
        CheckNdArray(m12,
                     "[[[1, 3, 5]],\n"
                     " [[7, 9, 11]]]");
    }

    SECTION("Matmul (ND, MD)") {
        auto m1 = NdArray::Arange(36.f).reshape(2, 3, 1, 2, 3);
        auto m2 = NdArray::Arange(36.f).reshape(3, 3, 4);
        auto m12 = Matmul(m1, m2);
        CHECK(m12.shape() == Shape{2, 3, 3, 2, 4});
        CheckNdArray(m12,
                     "[[[[[20, 23, 26, 29],\n"
                     "    [56, 68, 80, 92]],\n"
                     "   [[56, 59, 62, 65],\n"
                     "    [200, 212, 224, 236]],\n"
                     "   [[92, 95, 98, 101],\n"
                     "    [344, 356, 368, 380]]],\n"
                     "  [[[92, 113, 134, 155],\n"
                     "    [128, 158, 188, 218]],\n"
                     "   [[344, 365, 386, 407],\n"
                     "    [488, 518, 548, 578]],\n"
                     "   [[596, 617, 638, 659],\n"
                     "    [848, 878, 908, 938]]],\n"
                     "  [[[164, 203, 242, 281],\n"
                     "    [200, 248, 296, 344]],\n"
                     "   [[632, 671, 710, 749],\n"
                     "    [776, 824, 872, 920]],\n"
                     "   [[1100, 1139, 1178, 1217],\n"
                     "    [1352, 1400, 1448, 1496]]]],\n"
                     " [[[[236, 293, 350, 407],\n"
                     "    [272, 338, 404, 470]],\n"
                     "   [[920, 977, 1034, 1091],\n"
                     "    [1064, 1130, 1196, 1262]],\n"
                     "   [[1604, 1661, 1718, 1775],\n"
                     "    [1856, 1922, 1988, 2054]]],\n"
                     "  [[[308, 383, 458, 533],\n"
                     "    [344, 428, 512, 596]],\n"
                     "   [[1208, 1283, 1358, 1433],\n"
                     "    [1352, 1436, 1520, 1604]],\n"
                     "   [[2108, 2183, 2258, 2333],\n"
                     "    [2360, 2444, 2528, 2612]]],\n"
                     "  [[[380, 473, 566, 659],\n"
                     "    [416, 518, 620, 722]],\n"
                     "   [[1496, 1589, 1682, 1775],\n"
                     "    [1640, 1742, 1844, 1946]],\n"
                     "   [[2612, 2705, 2798, 2891],\n"
                     "    [2864, 2966, 3068, 3170]]]]]");
    }

    // ----------------------------- Cross product -----------------------------
    SECTION("Cross (1D, 1D), (3, 3 elem)") {
        NdArray m1 = {1.f, 2.f, 3.f};
        NdArray m2 = {4.f, 5.f, 6.f};
        auto m12 = m1.cross(m2);
        CheckNdArray(m12, "[-3, 6, -3]");
    }

    SECTION("Cross (1D, 1D), (3, 2 elem)") {
        NdArray m1 = {1.f, 2.f, 3.f};
        NdArray m2 = {4.f, 5.f};
        auto m12 = m1.cross(m2);
        CheckNdArray(m12, "[-15, 12, -3]");
    }

    SECTION("Cross (1D, 1D), (2, 3 elem)") {
        NdArray m1 = {1.f, 2.f};
        NdArray m2 = {4.f, 5.f, 6.f};
        auto m12 = m1.cross(m2);
        CheckNdArray(m12, "[12, -6, -3]");
    }

    SECTION("Cross (1D, 1D), (2, 2 elem)") {
        NdArray m1 = {1.f, 2.f};
        NdArray m2 = {4.f, 5.f};
        float m12 = m1.cross(m2);
        CHECK(m12 == Approx(-3.f));
    }

    SECTION("Cross (1D, 1D), (mismatch)") {
        NdArray m1 = {1.f};
        NdArray m2 = {4.f, 5.f};
        NdArray m3 = {4.f, 5.f, 6.f, 7.f};
        CHECK_THROWS(m1.cross(m2));
        CHECK_THROWS(m2.cross(m3));
    }

    SECTION("Cross (ND, MD), (3, 3 elem)") {
        auto m1 = NdArray::Arange(18.f).reshape(3, 2, 3);
        auto m2 = NdArray::Arange(6.f).reshape(2, 3) + 1.f;
        auto m12 = m1.cross(m2);
        CheckNdArray(m12,
                     "[[[-1, 2, -1],\n"
                     "  [-1, 2, -1]],\n"
                     " [[5, -10, 5],\n"
                     "  [5, -10, 5]],\n"
                     " [[11, -22, 11],\n"
                     "  [11, -22, 11]]]");
    }

    SECTION("Cross (ND, MD), (3, 2 elem)") {
        auto m1 = NdArray::Arange(18.f).reshape(2, 3, 3);
        auto m2 = NdArray::Arange(6.f).reshape(3, 2) + 1.f;
        auto m12 = m1.cross(m2);
        CheckNdArray(m12,
                     "[[[-4, 2, -1],\n"
                     "  [-20, 15, 0],\n"
                     "  [-48, 40, 1]],\n"
                     " [[-22, 11, 8],\n"
                     "  [-56, 42, 9],\n"
                     "  [-102, 85, 10]]]");
    }

    SECTION("Cross (ND, MD), (2, 3 elem)") {
        auto m1 = NdArray::Arange(12.f).reshape(3, 2, 2);
        auto m2 = NdArray::Arange(6.f).reshape(2, 3) + 1.f;
        auto m12 = m1.cross(m2);
        CheckNdArray(m12,
                     "[[[3, -0, -1],\n"
                     "  [18, -12, -2]],\n"
                     " [[15, -12, 3],\n"
                     "  [42, -36, 2]],\n"
                     " [[27, -24, 7],\n"
                     "  [66, -60, 6]]]");
    }

    SECTION("Cross (ND, MD), (2, 2 elem)") {
        auto m1 = NdArray::Arange(12.f).reshape(3, 2, 2);
        auto m2 = NdArray::Arange(4.f).reshape(2, 2) + 1.f;
        auto m12 = m1.cross(m2);
        CheckNdArray(m12,
                     "[[-1, -1],\n"
                     " [3, 3],\n"
                     " [7, 7]]");
    }

    // ----------------------------- Axis operation ----------------------------
    SECTION("Sum") {
        NdArray m0;
        auto m1 = NdArray::Arange(6.f);
        auto m2 = NdArray::Arange(36.f).reshape(2, 3, 2, 3);
        CheckNdArray(m0.sum(), "[0]");
        CheckNdArray(m1.sum(), "[15]");
        CheckNdArray(m2.sum({0}),
                     "[[[18, 20, 22],\n"
                     "  [24, 26, 28]],\n"
                     " [[30, 32, 34],\n"
                     "  [36, 38, 40]],\n"
                     " [[42, 44, 46],\n"
                     "  [48, 50, 52]]]");
        CheckNdArray(m2.sum({2}),
                     "[[[3, 5, 7],\n"
                     "  [15, 17, 19],\n"
                     "  [27, 29, 31]],\n"
                     " [[39, 41, 43],\n"
                     "  [51, 53, 55],\n"
                     "  [63, 65, 67]]]");
        CheckNdArray(m2.sum({3}),
                     "[[[3, 12],\n"
                     "  [21, 30],\n"
                     "  [39, 48]],\n"
                     " [[57, 66],\n"
                     "  [75, 84],\n"
                     "  [93, 102]]]");
        CheckNdArray(m2.sum({1, 2}),
                     "[[45, 51, 57],\n"
                     " [153, 159, 165]]");
        CheckNdArray(m2.sum({1, 3}),
                     "[[63, 90],\n"
                     " [225, 252]]");
        CheckNdArray(m2.sum({0, 1, 2, 3}), "[630]");
        CheckNdArray(m1.reshape(1, 2, 3).sum({0, 1, 2}), "[15]");
    }

    SECTION("Min") {
        NdArray m0;
        auto m1 = NdArray::Arange(6.f) - 3.f;
        auto m2 = NdArray::Arange(12.f).reshape(2, 3, 2) - 6.f;
        CHECK_THROWS(m0.min());
        CheckNdArray(m1.min(), "[-3]");
        CheckNdArray(m2.min({0}),
                     "[[-6, -5],\n"
                     " [-4, -3],\n"
                     " [-2, -1]]");
        CheckNdArray(m2.min({2}),
                     "[[-6, -4, -2],\n"
                     " [0, 2, 4]]");
        CheckNdArray(m2.min({2, 1}), "[-6, 0]");
    }

    SECTION("Max") {
        NdArray m0;
        auto m1 = NdArray::Arange(6.f) - 3.f;
        auto m2 = NdArray::Arange(12.f).reshape(2, 3, 2) - 6.f;
        CHECK_THROWS(m0.max());
        CheckNdArray(m1.max(), "[2]");
        CheckNdArray(m2.max({0}),
                     "[[0, 1],\n"
                     " [2, 3],\n"
                     " [4, 5]]");
        CheckNdArray(m2.max({2}),
                     "[[-5, -3, -1],\n"
                     " [1, 3, 5]]");
        CheckNdArray(m2.max({2, 1}), "[-1, 5]");
    }

    SECTION("Mean") {
        NdArray m0;
        auto m1 = NdArray::Arange(6.f) - 3.f;
        auto m2 = NdArray::Arange(12.f).reshape(2, 3, 2) - 6.f;
        CheckNdArray(m0.mean(), "[nan]");
        CheckNdArray(m1.mean(), "[-0.5]");
        CheckNdArray(m2.mean({0}),
                     "[[-3, -2],\n"
                     " [-1, 0],\n"
                     " [1, 2]]");
        CheckNdArray(m2.mean({2}),
                     "[[-5.5, -3.5, -1.5],\n"
                     " [0.5, 2.5, 4.5]]");
        CheckNdArray(m2.mean({2, 1}), "[-3.5, 2.5]");
    }

    SECTION("Sum keepdims") {
        NdArray m0;
        auto m1 = NdArray::Arange(36.f).reshape(2, 3, 2, 3);
        CheckNdArray(m1.sum({1, 2}, true),
                     "[[[[45, 51, 57]]],\n"
                     " [[[153, 159, 165]]]]");
        CheckNdArray(m1.sum({1, 3}, true),
                     "[[[[63],\n"
                     "   [90]]],\n"
                     " [[[225],\n"
                     "   [252]]]]");
        CheckNdArray(m1.sum({0, 1, 2, 3}, true), "[[[[630]]]]");
        CheckNdArray(m1.sum({}, true), "[[[[630]]]]");
    }

    // --------------------------- Logistic operation --------------------------
    SECTION("All (no axis)") {
        NdArray m0;
        CHECK(All(m0));
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = m1.copy();
        CHECK(All(m1 == m2));
        m2(1, 2) = -1.f;
        CHECK(!All(m1 == m2));
    }

    SECTION("Any (no axis)") {
        NdArray m0;
        CHECK(!Any(m0));
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = m1.copy() + 1.f;
        CHECK(!Any(m1 == m2));
        m2(0, 0) = 0.f;
        CHECK(Any(m1 == m2));
    }

    SECTION("All (with axis)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        CheckNdArray(All(m1, {0}), "[0, 1, 1]");
        CheckNdArray(All(m1, {1}), "[0, 1]");
    }

    SECTION("Any (with axis)") {
        auto m1 = NdArray::Zeros(2, 3);
        m1(0, 0) = -1.f;
        CheckNdArray(Any(m1, {0}), "[1, 0, 0]");
        CheckNdArray(Any(m1, {1}), "[1, 0]");
    }

    SECTION("Where") {
        auto m1 = 2.f < NdArray::Arange(6.f).reshape(2, 3);
        CheckNdArray(Where(m1, NdArray::Ones(2, 1), NdArray::Arange(3)),
                     "[[0, 1, 2],\n"
                     " [1, 1, 1]]");
        CheckNdArray(Where(m1, 1.f, NdArray::Arange(3)),
                     "[[0, 1, 2],\n"
                     " [1, 1, 1]]");
        CheckNdArray(Where(m1, NdArray::Arange(3), 0.f),
                     "[[0, 0, 0],\n"
                     " [0, 1, 2]]");
        CheckNdArray(Where(m1, 1.f, 0.f),
                     "[[0, 0, 0],\n"
                     " [1, 1, 1]]");
        CheckNdArray(Where(m1, 0.f, 1.f),
                     "[[1, 1, 1],\n"
                     " [0, 0, 0]]");
    }

    // ------------------------------- Operator --------------------------------
    SECTION("Single +- operators") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m_p = +m1;
        auto m_n = -m1;
        CHECK(m_p.id() != m1.id());
        CHECK(m_n.id() != m1.id());
        CheckNdArray(m_p,
                     "[[0, 1, 2],\n"
                     " [3, 4, 5]]");
        CheckNdArray(m_n,
                     "[[-0, -1, -2],\n"
                     " [-3, -4, -5]]");
    }

    SECTION("Add same shape") {
        auto m1 = NdArray::Arange(12.f).reshape(2, 3, 2);
        auto m2 = NdArray::Ones({2, 3, 2});
        auto m3 = m1 + m2;
        CHECK(m1.shape() == m2.shape());
        CHECK(m1.shape() == m3.shape());
        CheckNdArray(m3,
                     "[[[1, 2],\n"
                     "  [3, 4],\n"
                     "  [5, 6]],\n"
                     " [[7, 8],\n"
                     "  [9, 10],\n"
                     "  [11, 12]]]");
    }

    SECTION("Add broadcast 2-dim") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(2.f).reshape(2, 1);
        auto m3 = NdArray::Arange(3.f).reshape(1, 3);
        auto m12 = m1 + m2;
        auto m13 = m1 + m3;
        auto m23 = m2 + m3;
        CHECK(m12.shape() == Shape{2, 3});
        CHECK(m13.shape() == Shape{2, 3});
        CHECK(m23.shape() == Shape{2, 3});
        CheckNdArray(m12,
                     "[[0, 1, 2],\n"
                     " [4, 5, 6]]");
        CheckNdArray(m13,
                     "[[0, 2, 4],\n"
                     " [3, 5, 7]]");
        CheckNdArray(m23,
                     "[[0, 1, 2],\n"
                     " [1, 2, 3]]");
    }

    SECTION("Add broadcast high-dim") {
        auto m1 = NdArray::Arange(6.f).reshape(1, 2, 1, 1, 3);
        auto m2 = NdArray::Arange(2.f).reshape(2, 1);
        auto m3 = NdArray::Arange(3.f).reshape(1, 3);
        auto m12 = m1 + m2;
        auto m13 = m1 + m3;
        CHECK(m12.shape() == Shape{1, 2, 1, 2, 3});
        CHECK(m13.shape() == Shape{1, 2, 1, 1, 3});
        CheckNdArray(m12,
                     "[[[[[0, 1, 2],\n"
                     "    [1, 2, 3]]],\n"
                     "  [[[3, 4, 5],\n"
                     "    [4, 5, 6]]]]]");
        CheckNdArray(m13,
                     "[[[[[0, 2, 4]]],\n"
                     "  [[[3, 5, 7]]]]]");
    }

    SECTION("Add empty") {
        NdArray m1;
        auto m2 = m1 + 1.f;
        CHECK(m1.shape() == Shape{0});
        CHECK(m2.shape() == Shape{0});
        CHECK(m2.size() == 0);
    }

    SECTION("Sub/Mul/Div") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(3.f).reshape(3);
        auto m_sub = m1 - m2;
        auto m_mul = m1 * m2;
        auto m_div = m1 / m2;
        CHECK(m_sub.shape() == Shape{2, 3});
        CHECK(m_mul.shape() == Shape{2, 3});
        CHECK(m_div.shape() == Shape{2, 3});
        CheckNdArray(m_sub,
                     "[[0, 0, 0],\n"
                     " [3, 3, 3]]");
        CheckNdArray(m_mul,
                     "[[0, 1, 4],\n"
                     " [0, 4, 10]]");
        ResolveAmbiguous(m_div);  // -nan -> nan
        CheckNdArray(m_div,
                     "[[nan, 1, 1],\n"
                     " [inf, 4, 2.5]]");
    }

    SECTION("Arithmetic operators (NdArray, float)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m_add = m1 + 10.f;
        auto m_sub = m1 - 10.f;
        auto m_mul = m1 * 10.f;
        auto m_div = m1 / 10.f;
        CheckNdArray(m_add,
                     "[[10, 11, 12],\n"
                     " [13, 14, 15]]");
        CheckNdArray(m_sub,
                     "[[-10, -9, -8],\n"
                     " [-7, -6, -5]]");
        CheckNdArray(m_mul,
                     "[[0, 10, 20],\n"
                     " [30, 40, 50]]");
        CheckNdArray(m_div,
                     "[[0, 0.1, 0.2],\n"
                     " [0.3, 0.4, 0.5]]");
    }

    SECTION("Arithmetic operators (float, NdArray)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m_add = 10.f + m1;
        auto m_sub = 10.f - m1;
        auto m_mul = 10.f * m1;
        auto m_div = 10.f / m1;
        CheckNdArray(m_add,
                     "[[10, 11, 12],\n"
                     " [13, 14, 15]]");
        CheckNdArray(m_sub,
                     "[[10, 9, 8],\n"
                     " [7, 6, 5]]");
        CheckNdArray(m_mul,
                     "[[0, 10, 20],\n"
                     " [30, 40, 50]]");
        CheckNdArray(m_div,
                     "[[inf, 10, 5],\n"
                     " [3.33333, 2.5, 2]]");
    }

    SECTION("Comparison operators (NdArray, NdArray)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(2.f).reshape(2, 1) + 3.f;
        CheckNdArray(m1 == m2,
                     "[[0, 0, 0],\n"
                     " [0, 1, 0]]");
        CheckNdArray(m1 != m2,
                     "[[1, 1, 1],\n"
                     " [1, 0, 1]]");
        CheckNdArray(m1 > m2,
                     "[[0, 0, 0],\n"
                     " [0, 0, 1]]");
        CheckNdArray(m1 >= m2,
                     "[[0, 0, 0],\n"
                     " [0, 1, 1]]");
        CheckNdArray(m1 < m2,
                     "[[1, 1, 1],\n"
                     " [1, 0, 0]]");
        CheckNdArray(m1 <= m2,
                     "[[1, 1, 1],\n"
                     " [1, 1, 0]]");
    }

    SECTION("Comparison operators (NdArray, float)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        CheckNdArray(m1 == 1.f,
                     "[[0, 1, 0],\n"
                     " [0, 0, 0]]");
        CheckNdArray(m1 != 1.f,
                     "[[1, 0, 1],\n"
                     " [1, 1, 1]]");
        CheckNdArray(m1 > 1.f,
                     "[[0, 0, 1],\n"
                     " [1, 1, 1]]");
        CheckNdArray(m1 >= 1.f,
                     "[[0, 1, 1],\n"
                     " [1, 1, 1]]");
        CheckNdArray(m1 < 1.f,
                     "[[1, 0, 0],\n"
                     " [0, 0, 0]]");
        CheckNdArray(m1 <= 1.f,
                     "[[1, 1, 0],\n"
                     " [0, 0, 0]]");
    }

    SECTION("Comparison operators (float, NdArray)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        CheckNdArray(1.f == m1,
                     "[[0, 1, 0],\n"
                     " [0, 0, 0]]");
        CheckNdArray(1.f != m1,
                     "[[1, 0, 1],\n"
                     " [1, 1, 1]]");
        CheckNdArray(1.f > m1,
                     "[[1, 0, 0],\n"
                     " [0, 0, 0]]");
        CheckNdArray(1.f >= m1,
                     "[[1, 1, 0],\n"
                     " [0, 0, 0]]");
        CheckNdArray(1.f < m1,
                     "[[0, 0, 1],\n"
                     " [1, 1, 1]]");
        CheckNdArray(1.f <= m1,
                     "[[0, 1, 1],\n"
                     " [1, 1, 1]]");
    }

    // --------------------------- In-place Operator ---------------------------
    SECTION("Arithmetic operators (NdArray, NdArray) (in-place both)") {
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f),
                "[[0, 2, 4],\n"
                " [3, 5, 7]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator+));
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f),
                "[[0, 0, 0],\n"
                " [3, 3, 3]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator-));
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f),
                "[[0, 1, 4],\n"
                " [0, 4, 10]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator*));
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f) + 1.f,
                "[[0, 0.5, 0.666667],\n"
                " [3, 2, 1.66667]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator/));
    }

    SECTION("Arithmetic operators (NdArray, NdArray) (in-place right)") {
        auto m1 = NdArray::Arange(3.f);
        auto m2 = NdArray::Arange(3.f) + 1.f;
        CheckNdArrayInplace(
                m1, NdArray::Arange(6.f).reshape(2, 3),
                "[[0, 2, 4],\n"
                " [3, 5, 7]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(operator+));
        CheckNdArrayInplace(
                m1, NdArray::Arange(6.f).reshape(2, 3),
                "[[0, 0, 0],\n"
                " [-3, -3, -3]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(operator-));
        CheckNdArrayInplace(
                m1, NdArray::Arange(6.f).reshape(2, 3),
                "[[0, 1, 4],\n"
                " [0, 4, 10]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(operator*));
        CheckNdArrayInplace(
                m2, NdArray::Arange(6.f).reshape(2, 3),
                "[[inf, 2, 1.5],\n"
                " [0.333333, 0.5, 0.6]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(operator/));
    }

    SECTION("Arithmetic operators (NdArray, NdArray) (in-place left)") {
        auto m2 = NdArray::Arange(3.f);
        auto m3 = NdArray::Arange(3.f) + 1.f;
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m2,
                "[[0, 2, 4],\n"
                " [3, 5, 7]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(operator+));
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m2,
                "[[0, 0, 0],\n"
                " [3, 3, 3]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(operator-));
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m2,
                "[[0, 1, 4],\n"
                " [0, 4, 10]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(operator*));
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m3,
                "[[0, 0.5, 0.666667],\n"
                " [3, 2, 1.66667]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(operator/));
    }

    SECTION("Arithmetic operators (NdArray, float) (inplace)") {
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 2.f, "[2, 3, 4]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator+));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 2.f, "[-2, -1, 0]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator-));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 2.f, "[0, 2, 4]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator*));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 2.f, "[0, 0.5, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator/));
    }

    SECTION("Arithmetic operators (float, NdArray) (inplace)") {
        CheckNdArrayInplace(
                2.f, NdArray::Arange(3.f), "[2, 3, 4]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator+));
        CheckNdArrayInplace(
                2.f, NdArray::Arange(3.f), "[2, 1, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator-));
        CheckNdArrayInplace(
                2.f, NdArray::Arange(3.f), "[0, 2, 4]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator*));
        CheckNdArrayInplace(
                2.f, NdArray::Arange(3.f), "[inf, 2, 1]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator/));
    }

    SECTION("Comparison operators (NdArray, NdArray) (inplace both)") {
        CheckNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[0, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator==));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[1, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator!=));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[0, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator>));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[0, 1, 1]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator>=));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[1, 0, 0]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator<));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[1, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator<=));
    }

    SECTION("Comparison operators (NdArray, NdArray) (inplace right)") {
        auto m2 = NdArray::Zeros(1) + 1.f;
        CheckNdArrayInplace(NdArray::Arange(3.f), m2, "[0, 1, 0]",
                            static_cast<NdArray (*)(NdArray&&, const NdArray&)>(
                                    operator==));
        CheckNdArrayInplace(NdArray::Arange(3.f), m2, "[1, 0, 1]",
                            static_cast<NdArray (*)(NdArray&&, const NdArray&)>(
                                    operator!=));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), m2, "[0, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(operator>));
        CheckNdArrayInplace(NdArray::Arange(3.f), m2, "[0, 1, 1]",
                            static_cast<NdArray (*)(NdArray&&, const NdArray&)>(
                                    operator>=));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), m2, "[1, 0, 0]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(operator<));
        CheckNdArrayInplace(NdArray::Arange(3.f), m2, "[1, 1, 0]",
                            static_cast<NdArray (*)(NdArray&&, const NdArray&)>(
                                    operator<=));
    }

    SECTION("Comparison operators (NdArray, NdArray) (inplace left)") {
        auto m1 = NdArray::Zeros(1) + 1.f;
        CheckNdArrayInplace(m1, NdArray::Arange(3.f), "[0, 1, 0]",
                            static_cast<NdArray (*)(const NdArray&, NdArray&&)>(
                                    operator==));
        CheckNdArrayInplace(m1, NdArray::Arange(3.f), "[1, 0, 1]",
                            static_cast<NdArray (*)(const NdArray&, NdArray&&)>(
                                    operator!=));
        CheckNdArrayInplace(
                m1, NdArray::Arange(3.f), "[1, 0, 0]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(operator>));
        CheckNdArrayInplace(m1, NdArray::Arange(3.f), "[1, 1, 0]",
                            static_cast<NdArray (*)(const NdArray&, NdArray&&)>(
                                    operator>=));
        CheckNdArrayInplace(
                m1, NdArray::Arange(3.f), "[0, 0, 1]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(operator<));
        CheckNdArrayInplace(m1, NdArray::Arange(3.f), "[0, 1, 1]",
                            static_cast<NdArray (*)(const NdArray&, NdArray&&)>(
                                    operator<=));
    }

    SECTION("Comparison operators (NdArray, float) (inplace)") {
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[0, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator==));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[1, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator!=));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[0, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator>));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[0, 1, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator>=));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[1, 0, 0]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator<));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[1, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator<=));
    }

    SECTION("Comparison operators (NdArray, float) (inplace)") {
        CheckNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[0, 1, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator==));
        CheckNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[1, 0, 1]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator!=));
        CheckNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[1, 0, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator>));
        CheckNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[1, 1, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator>=));
        CheckNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[0, 0, 1]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator<));
        CheckNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[0, 1, 1]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator<=));
    }

    SECTION("Compound assignment operators (NdArray, NdArray) (in-plafce, &)") {
        auto m0 = NdArray::Arange(6.f).reshape(2, 3);
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(6.f).reshape(2, 3);
        auto m3 = NdArray::Arange(6.f).reshape(2, 3);
        auto m4 = NdArray::Arange(6.f).reshape(2, 3);
        auto m5 = NdArray::Arange(3.f);
        auto m0_id = m0.id();
        auto m1_id = m1.id();
        auto m2_id = m2.id();
        auto m3_id = m3.id();
        auto m4_id = m4.id();
        m0 += m0;
        m1 += NdArray::Arange(3.f);
        m2 -= NdArray::Arange(3.f);
        m3 *= NdArray::Arange(3.f);
        m4 /= NdArray::Arange(3.f);
        CheckNdArray(m0,
                     "[[0, 2, 4],\n"
                     " [6, 8, 10]]");
        CheckNdArray(m1,
                     "[[0, 2, 4],\n"
                     " [3, 5, 7]]");
        CheckNdArray(m2,
                     "[[0, 0, 0],\n"
                     " [3, 3, 3]]");
        CheckNdArray(m3,
                     "[[0, 1, 4],\n"
                     " [0, 4, 10]]");
        ResolveAmbiguous(m4);  // -nan -> nan
        CheckNdArray(m4,
                     "[[nan, 1, 1],\n"
                     " [inf, 4, 2.5]]");
        CHECK(m0.id() == m0_id);  // in-place
        CHECK(m1.id() == m1_id);
        CHECK(m2.id() == m2_id);
        CHECK(m3.id() == m3_id);
        CHECK(m4.id() == m4_id);
        // size change is not allowed
        CHECK_THROWS(m5 += m0);
        auto m6 = m0.reshape(2, 1, 3);
        CHECK_THROWS(m6 *= m5.reshape(3, 1));
    }

    SECTION("Compound assignment operators (NdArray, float) (in-plafce, &)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(6.f).reshape(2, 3);
        auto m3 = NdArray::Arange(6.f).reshape(2, 3);
        auto m4 = NdArray::Arange(6.f).reshape(2, 3);
        auto m1_id = m1.id();
        auto m2_id = m2.id();
        auto m3_id = m3.id();
        auto m4_id = m4.id();
        m1 += 10.f;
        m2 -= 10.f;
        m3 *= 10.f;
        m4 /= 10.f;
        CheckNdArray(m1,
                     "[[10, 11, 12],\n"
                     " [13, 14, 15]]");
        CheckNdArray(m2,
                     "[[-10, -9, -8],\n"
                     " [-7, -6, -5]]");
        CheckNdArray(m3,
                     "[[0, 10, 20],\n"
                     " [30, 40, 50]]");
        CheckNdArray(m4,
                     "[[0, 0.1, 0.2],\n"
                     " [0.3, 0.4, 0.5]]");
        CHECK(m1.id() == m1_id);
        CHECK(m2.id() == m2_id);
        CHECK(m3.id() == m3_id);
        CHECK(m4.id() == m4_id);
    }

    SECTION("Compound assignment operators (in-plafce, &&)") {
        // (NdArray, NdArray)
        auto m1 = NdArray::Arange(6.f).reshape(2, 3) += NdArray::Arange(3.f);
        auto m2 = NdArray::Arange(6.f).reshape(2, 3) -= NdArray::Arange(3.f);
        auto m3 = NdArray::Arange(6.f).reshape(2, 3) *= NdArray::Arange(3.f);
        auto m4 = NdArray::Arange(6.f).reshape(2, 3) /= NdArray::Arange(3.f);
        CheckNdArray(m1,
                     "[[0, 2, 4],\n"
                     " [3, 5, 7]]");
        CheckNdArray(m2,
                     "[[0, 0, 0],\n"
                     " [3, 3, 3]]");
        CheckNdArray(m3,
                     "[[0, 1, 4],\n"
                     " [0, 4, 10]]");
        ResolveAmbiguous(m4);  // -nan -> nan
        CheckNdArray(m4,
                     "[[nan, 1, 1],\n"
                     " [inf, 4, 2.5]]");
        // (NdArray, float)
        auto m5 = NdArray::Arange(6.f).reshape(2, 3) += 10.f;
        auto m6 = NdArray::Arange(6.f).reshape(2, 3) -= 10.f;
        auto m7 = NdArray::Arange(6.f).reshape(2, 3) *= 10.f;
        auto m8 = NdArray::Arange(6.f).reshape(2, 3) /= 10.f;
        CheckNdArray(m5,
                     "[[10, 11, 12],\n"
                     " [13, 14, 15]]");
        CheckNdArray(m6,
                     "[[-10, -9, -8],\n"
                     " [-7, -6, -5]]");
        CheckNdArray(m7,
                     "[[0, 10, 20],\n"
                     " [30, 40, 50]]");
        CheckNdArray(m8,
                     "[[0, 0.1, 0.2],\n"
                     " [0.3, 0.4, 0.5]]");
    }

    // --------------------------- Operator function ---------------------------
    SECTION("Single") {
        auto m1 = NdArray::Arange(3.f);
        auto m2 = Positive(m1);
        auto m3 = Negative(m1);
        m1[0] = -1.f;
        CheckNdArray(m1, "[-1, 1, 2]");
        CheckNdArray(m2, "[0, 1, 2]");
        CheckNdArray(m3, "[-0, -1, -2]");
    }

    SECTION("Function Arithmetic (NdArray, NdArray)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(3.f);
        auto m_add = Add(m1, m2);
        auto m_sub = Subtract(m1, m2);
        auto m_mul = Multiply(m1, m2);
        auto m_div = Divide(m1, m2);
        CheckNdArray(m_add,
                     "[[0, 2, 4],\n"
                     " [3, 5, 7]]");
        CheckNdArray(m_sub,
                     "[[0, 0, 0],\n"
                     " [3, 3, 3]]");
        CheckNdArray(m_mul,
                     "[[0, 1, 4],\n"
                     " [0, 4, 10]]");
        ResolveAmbiguous(m_div);  // -nan -> nan
        CheckNdArray(m_div,
                     "[[nan, 1, 1],\n"
                     " [inf, 4, 2.5]]");
    }

    SECTION("Function Arithmetic (NdArray, float)") {
        auto m1 = NdArray::Arange(3.f);
        auto m_add = Add(m1, 2.f);
        auto m_sub = Subtract(m1, 2.f);
        auto m_mul = Multiply(m1, 2.f);
        auto m_div = Divide(m1, 2.f);
        CheckNdArray(m_add, "[2, 3, 4]");
        CheckNdArray(m_sub, "[-2, -1, 0]");
        CheckNdArray(m_mul, "[0, 2, 4]");
        CheckNdArray(m_div, "[0, 0.5, 1]");
    }

    SECTION("Function Arithmetic (float, NdArray)") {
        auto m1 = NdArray::Arange(3.f);
        auto m_add = Add(2.f, m1);
        auto m_sub = Subtract(2.f, m1);
        auto m_mul = Multiply(2.f, m1);
        auto m_div = Divide(2.f, m1);
        CheckNdArray(m_add, "[2, 3, 4]");
        CheckNdArray(m_sub, "[2, 1, 0]");
        CheckNdArray(m_mul, "[0, 2, 4]");
        CheckNdArray(m_div, "[inf, 2, 1]");
    }

    SECTION("Function Comparison") {
        auto m1 = NdArray::Arange(3.f);
        auto m2 = NdArray::Zeros(1) + 1.f;
        CheckNdArray(Equal(m1, m2), "[0, 1, 0]");
        CheckNdArray(NotEqual(m1, m2), "[1, 0, 1]");
        CheckNdArray(Greater(m1, m2), "[0, 0, 1]");
        CheckNdArray(GreaterEqual(m1, m2), "[0, 1, 1]");
        CheckNdArray(Less(m1, m2), "[1, 0, 0]");
        CheckNdArray(LessEqual(m1, m2), "[1, 1, 0]");
        CheckNdArray(Equal(m1, 1.f), "[0, 1, 0]");
        CheckNdArray(NotEqual(m1, 1.f), "[1, 0, 1]");
        CheckNdArray(Greater(m1, 1.f), "[0, 0, 1]");
        CheckNdArray(GreaterEqual(m1, 1.f), "[0, 1, 1]");
        CheckNdArray(Less(m1, 1.f), "[1, 0, 0]");
        CheckNdArray(LessEqual(m1, 1.f), "[1, 1, 0]");
        CheckNdArray(Equal(1.f, m1), "[0, 1, 0]");
        CheckNdArray(NotEqual(1.f, m1), "[1, 0, 1]");
        CheckNdArray(Greater(1.f, m1), "[1, 0, 0]");
        CheckNdArray(GreaterEqual(1.f, m1), "[1, 1, 0]");
        CheckNdArray(Less(1.f, m1), "[0, 0, 1]");
        CheckNdArray(LessEqual(1.f, m1), "[0, 1, 1]");
    }

    SECTION("Function Dot") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(3.f);
        auto m12 = Dot(m1, m2);
        auto m_a = Dot(m2, 2.f);
        auto m_b = Dot(2.f, m2);
        CheckNdArray(m12, "[5, 14]");
        CheckNdArray(m_a, "[0, 2, 4]");
        CheckNdArray(m_b, "[0, 2, 4]");
    }

    SECTION("Function Cross") {
        NdArray m1 = {1.f, 2.f, 3.f};
        NdArray m2 = {4.f, 5.f, 6.f};
        auto m12 = Cross(m1, m2);
        CheckNdArray(m12, "[-3, 6, -3]");
    }

    SECTION("Function Basic Math") {
        auto m1 = NdArray::Arange(3.f);
        auto m2 = NdArray::Arange(7.f) / 3.f - 1.f;
        CheckNdArray(-m1, "[-0, -1, -2]");
        CheckNdArray(Abs(-m1), "[0, 1, 2]");
        CheckNdArray(Sign(m2), "[-1, -1, -1, 0, 1, 1, 1]");
        CheckNdArray(Ceil(m2), "[-1, -0, -0, 0, 1, 1, 1]");
        CheckNdArray(Floor(m2), "[-1, -1, -1, 0, 0, 0, 1]");
        CheckNdArray(Clip(m2, -0.5, 0.4),
                     "[-0.5, -0.5, -0.333333, 0, 0.333333, 0.4, 0.4]");
        CheckNdArray(Sqrt(m1), "[0, 1, 1.41421]");
        CheckNdArray(Exp(m1), "[1, 2.71828, 7.38906]");
        CheckNdArray(Log(m1), "[-inf, 0, 0.693147]");
        CheckNdArray(Square(m1), "[0, 1, 4]");
        CheckNdArray(Power(m1, m1 + 1.f), "[0, 1, 8]");
        CheckNdArray(Power(m1, 4.f), "[0, 1, 16]");
        CheckNdArray(Power(4.f, m1), "[1, 4, 16]");
    }

    SECTION("Function Trigonometric") {
        auto m1 = NdArray::Arange(3.f);
        CheckNdArray(Sin(m1), "[0, 0.841471, 0.909297]");
        CheckNdArray(Cos(m1), "[1, 0.540302, -0.416147]");
        CheckNdArray(Tan(m1), "[0, 1.55741, -2.18504]");
    }

    SECTION("Function Inverse-Trigonometric") {
        auto m1 = NdArray::Arange(3) - 1.f;
        auto m2 = NdArray::Arange(3) * 100.f;
        CheckNdArray(ArcSin(m1), "[-1.5708, 0, 1.5708]");
        CheckNdArray(ArcCos(m1), "[3.14159, 1.5708, 0]");
        CheckNdArray(ArcTan(m2), "[0, 1.5608, 1.5658]");
        CheckNdArray(ArcTan2(m1, m2), "[-1.5708, 0, 0.00499996]");
        CheckNdArray(ArcTan2(m1, 2.f), "[-0.463648, 0, 0.463648]");
        CheckNdArray(ArcTan2(2.f, m1), "[2.03444, 1.5708, 1.10715]");
    }

    SECTION("Function Axis") {
        auto m1 = NdArray::Arange(36.f).reshape(2, 3, 2, 3);
        CheckNdArray(Sum(m1, {1, 3}),
                     "[[63, 90],\n"
                     " [225, 252]]");
        auto m2 = NdArray::Arange(12.f).reshape(2, 3, 2) - 6.f;
        CheckNdArray(Min(m2, {2, 1}), "[-6, 0]");
        CheckNdArray(Max(m2, {2, 1}), "[-1, 5]");
        CheckNdArray(Mean(m2, {2, 1}), "[-3.5, 2.5]");
    }

    SECTION("Function Shape") {
        auto m1 = NdArray::Arange(6.f).reshape(1, 2, 1, 3, 1);
        CHECK(Reshape(m1, {2, 1, 3}).shape() == Shape{2, 1, 3});
        CHECK(Squeeze(m1).shape() == Shape{2, 3});
    }

    SECTION("Function Stack") {
        auto m1 = NdArray::Arange(12.f).reshape(4, 3);
        auto m2 = NdArray::Arange(12.f).reshape(4, 3) + 1.f;
        CheckNdArray(Stack({m1, m2}, 0),
                     "[[[0, 1, 2],\n"
                     "  [3, 4, 5],\n"
                     "  [6, 7, 8],\n"
                     "  [9, 10, 11]],\n"
                     " [[1, 2, 3],\n"
                     "  [4, 5, 6],\n"
                     "  [7, 8, 9],\n"
                     "  [10, 11, 12]]]");
        CheckNdArray(Stack({m1, m2}, 1),
                     "[[[0, 1, 2],\n"
                     "  [1, 2, 3]],\n"
                     " [[3, 4, 5],\n"
                     "  [4, 5, 6]],\n"
                     " [[6, 7, 8],\n"
                     "  [7, 8, 9]],\n"
                     " [[9, 10, 11],\n"
                     "  [10, 11, 12]]]");
        CheckNdArray(Stack({m1, m2}, 2),
                     "[[[0, 1],\n"
                     "  [1, 2],\n"
                     "  [2, 3]],\n"
                     " [[3, 4],\n"
                     "  [4, 5],\n"
                     "  [5, 6]],\n"
                     " [[6, 7],\n"
                     "  [7, 8],\n"
                     "  [8, 9]],\n"
                     " [[9, 10],\n"
                     "  [10, 11],\n"
                     "  [11, 12]]]");
        CHECK_THROWS(Stack({m1, m2}, -1));
        CHECK_THROWS(Stack({m1, m2}, 3));
    }

    SECTION("Function Concatenate") {
        auto m1 = NdArray::Arange(12.f).reshape(4, 3);
        auto m2 = NdArray::Arange(6.f).reshape(2, 3) + 1.f;
        auto m3 = NdArray::Arange(8.f).reshape(4, 2) + 1.f;
        CheckNdArray(Concatenate({m1, m2}, 0),
                     "[[0, 1, 2],\n"
                     " [3, 4, 5],\n"
                     " [6, 7, 8],\n"
                     " [9, 10, 11],\n"
                     " [1, 2, 3],\n"
                     " [4, 5, 6]]");
        CheckNdArray(Concatenate({m1, m3}, 1),
                     "[[0, 1, 2, 1, 2],\n"
                     " [3, 4, 5, 3, 4],\n"
                     " [6, 7, 8, 5, 6],\n"
                     " [9, 10, 11, 7, 8]]");
        CHECK_THROWS(Concatenate({m1, m2}, -1));
        CHECK_THROWS(Concatenate({m1, m2}, 1));
        CHECK_THROWS(Concatenate({m1, m2}, 2));
        CHECK_THROWS(Concatenate({m1, m3}, -1));
        CHECK_THROWS(Concatenate({m1, m3}, 0));
        CHECK_THROWS(Concatenate({m1, m3}, 2));

        auto m4 = NdArray::Arange(12.f).reshape(4, 1, 3);
        auto m5 = NdArray::Arange(6.f).reshape(2, 1, 3) + 1.f;
        auto m6 = NdArray::Arange(8.f).reshape(4, 1, 2) + 1.f;
        CheckNdArray(Concatenate({m4, m5}, 0),
                     "[[[0, 1, 2]],\n"
                     " [[3, 4, 5]],\n"
                     " [[6, 7, 8]],\n"
                     " [[9, 10, 11]],\n"
                     " [[1, 2, 3]],\n"
                     " [[4, 5, 6]]]");
        CheckNdArray(Concatenate({m4, m6}, 2),
                     "[[[0, 1, 2, 1, 2]],\n"
                     " [[3, 4, 5, 3, 4]],\n"
                     " [[6, 7, 8, 5, 6]],\n"
                     " [[9, 10, 11, 7, 8]]]");
        CHECK_THROWS(Concatenate({m4, m5}, -1));
        CHECK_THROWS(Concatenate({m4, m5}, 1));
        CHECK_THROWS(Concatenate({m4, m5}, 2));
        CHECK_THROWS(Concatenate({m4, m6}, -1));
        CHECK_THROWS(Concatenate({m4, m6}, 0));
        CHECK_THROWS(Concatenate({m4, m6}, 1));
    }

    SECTION("Function Split by indices") {
        auto m1 = NdArray::Arange(16.f).reshape(2, 4, 2);

        auto r0 = Split(m1, {1, 1}, 1);
        CHECK(r0.size() == 3);
        CheckNdArray(r0[0],
                     "[[[0, 1]],\n"
                     " [[8, 9]]]");
        CheckNdArray(r0[1], "[]");
        CheckNdArray(r0[2],
                     "[[[2, 3],\n"
                     "  [4, 5],\n"
                     "  [6, 7]],\n"
                     " [[10, 11],\n"
                     "  [12, 13],\n"
                     "  [14, 15]]]");

        auto r1 = Split(m1, {2, 0}, 1);
        CHECK(r1.size() == 3);
        CheckNdArray(r1[0],
                     "[[[0, 1],\n"
                     "  [2, 3]],\n"
                     " [[8, 9],\n"
                     "  [10, 11]]]");
        CheckNdArray(r1[1], "[]");
        CheckNdArray(r1[2],
                     "[[[0, 1],\n"
                     "  [2, 3],\n"
                     "  [4, 5],\n"
                     "  [6, 7]],\n"
                     " [[8, 9],\n"
                     "  [10, 11],\n"
                     "  [12, 13],\n"
                     "  [14, 15]]]");

        auto r2 = Split(m1, {0, 2, 3}, 1);
        CHECK(r2.size() == 4);
        CheckNdArray(r2[0], "[]");
        CheckNdArray(r2[1],
                     "[[[0, 1],\n"
                     "  [2, 3]],\n"
                     " [[8, 9],\n"
                     "  [10, 11]]]");
        CheckNdArray(r2[2],
                     "[[[4, 5]],\n"
                     " [[12, 13]]]");
        CheckNdArray(r2[3],
                     "[[[6, 7]],\n"
                     " [[14, 15]]]");

        auto r3 = Split(m1, {2, 4}, 1);
        CHECK(r3.size() == 3);
        CheckNdArray(r3[0],
                     "[[[0, 1],\n"
                     "  [2, 3]],\n"
                     " [[8, 9],\n"
                     "  [10, 11]]]");

        CheckNdArray(r3[1],
                     "[[[4, 5],\n"
                     "  [6, 7]],\n"
                     " [[12, 13],\n"
                     "  [14, 15]]]");

        CheckNdArray(r3[2], "[]");
    }

    SECTION("Function Split by n_section") {
        auto m1 = NdArray::Arange(16.f).reshape(2, 4, 2);
        auto r0 = Split(m1, 2, 1);
        CHECK(r0.size() == 2);
        CheckNdArray(r0[0],
                     "[[[0, 1],\n"
                     "  [2, 3]],\n"
                     " [[8, 9],\n"
                     "  [10, 11]]]");
        CheckNdArray(r0[1],
                     "[[[4, 5],\n"
                     "  [6, 7]],\n"
                     " [[12, 13],\n"
                     "  [14, 15]]]");

        auto r1 = Split(m1, 4, 1);
        CHECK(r1.size() == 4);
        CheckNdArray(r1[0],
                     "[[[0, 1]],\n"
                     " [[8, 9]]]");
        CheckNdArray(r1[1],
                     "[[[2, 3]],\n"
                     " [[10, 11]]]");
        CheckNdArray(r1[2],
                     "[[[4, 5]],\n"
                     " [[12, 13]]]");
        CheckNdArray(r1[3],
                     "[[[6, 7]],\n"
                     " [[14, 15]]]");

        auto r2 = Split(m1, 2, 2);
        CHECK(r2.size() == 2);
        CheckNdArray(r2[0],
                     "[[[0],\n"
                     "  [2],\n"
                     "  [4],\n"
                     "  [6]],\n"
                     " [[8],\n"
                     "  [10],\n"
                     "  [12],\n"
                     "  [14]]]");
        CheckNdArray(r2[1],
                     "[[[1],\n"
                     "  [3],\n"
                     "  [5],\n"
                     "  [7]],\n"
                     " [[9],\n"
                     "  [11],\n"
                     "  [13],\n"
                     "  [15]]]");
    }

    SECTION("Function Inverse (2d)") {
        auto m1 = NdArray::Arange(4).reshape(2, 2) + 1.f;
        auto m2 = Inv(m1);
        CheckNdArray(m2,
                     "[[-2, 1],\n"
                     " [1.5, -0.5]]",
                     4);  // Low precision
        CheckNdArray(m1.dot(m2),
                     "[[1, 0],\n"
                     " [0, 1]]",
                     4);  // Low precision
    }

    SECTION("Function Inverse (high-dim)") {
        auto m1 = NdArray::Arange(24).reshape(2, 3, 2, 2) + 1.f;
        auto m2 = Inv(m1);
        CheckNdArray(m2,
                     "[[[[-2, 1],\n"
                     "   [1.5, -0.5]],\n"
                     "  [[-4, 3],\n"
                     "   [3.5, -2.5]],\n"
                     "  [[-6, 5],\n"
                     "   [5.5, -4.5]]],\n"
                     " [[[-8, 7],\n"
                     "   [7.5, -6.5]],\n"
                     "  [[-10, 9],\n"
                     "   [9.5, -8.5]],\n"
                     "  [[-12, 11],\n"
                     "   [11.5, -10.5]]]]",
                     4);  // Low precision
    }

    // ----------------------- In-place Operator function ----------------------
    SECTION("Function in-place basic") {
        // In-place
        NdArray m1 = NdArray::Arange(3);
        NdArray m2 = NdArray::Arange(3);
        uintptr_t m1_id = m1.id();
        uintptr_t m2_id = m2.id();
        NdArray m3 = Power(std::move(m1), std::move(m2));
        CHECK((m3.id() == m1_id || m3.id() == m2_id));  // m3 is not new array
        // Not in-place
        CheckNdArrayNotInplace(-NdArray::Arange(3.f), "[0, 1, 2]",
                               static_cast<NdArray (*)(const NdArray&)>(Abs));
        // No matching broadcast
        NdArray m4 = NdArray::Arange(6).reshape(2, 1, 3);
        NdArray m5 = NdArray::Arange(2).reshape(2, 1);
        uintptr_t m4_id = m4.id();
        uintptr_t m5_id = m5.id();
        NdArray m6 = Power(std::move(m4), std::move(m5));
        CHECK((m6.id() != m4_id && m6.id() != m5_id));  // m6 is new array
    }

    SECTION("Single (inplace)") {
        CheckNdArrayInplace(NdArray::Arange(3.f), "[0, 1, 2]",
                            static_cast<NdArray (*)(NdArray &&)>(Positive));
        CheckNdArrayInplace(NdArray::Arange(3.f), "[-0, -1, -2]",
                            static_cast<NdArray (*)(NdArray &&)>(Negative));
    }

    SECTION("Function Arithmetic (NdArray, NdArray) (in-place both)") {
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f),
                "[[0, 2, 4],\n"
                " [3, 5, 7]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Add));
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f),
                "[[0, 0, 0],\n"
                " [3, 3, 3]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Subtract));
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f),
                "[[0, 1, 4],\n"
                " [0, 4, 10]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Multiply));
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f) + 1.f,
                "[[0, 0.5, 0.666667],\n"
                " [3, 2, 1.66667]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Divide));
    }

    SECTION("Function Arithmetic (NdArray, NdArray) (in-place right)") {
        auto m1 = NdArray::Arange(3.f);
        auto m2 = NdArray::Arange(3.f) + 1.f;
        CheckNdArrayInplace(
                m1, NdArray::Arange(6.f).reshape(2, 3),
                "[[0, 2, 4],\n"
                " [3, 5, 7]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Add));
        CheckNdArrayInplace(
                m1, NdArray::Arange(6.f).reshape(2, 3),
                "[[0, 0, 0],\n"
                " [-3, -3, -3]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Subtract));
        CheckNdArrayInplace(
                m1, NdArray::Arange(6.f).reshape(2, 3),
                "[[0, 1, 4],\n"
                " [0, 4, 10]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Multiply));
        CheckNdArrayInplace(
                m2, NdArray::Arange(6.f).reshape(2, 3),
                "[[inf, 2, 1.5],\n"
                " [0.333333, 0.5, 0.6]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Divide));
    }

    SECTION("Function Arithmetic (NdArray, NdArray) (in-place left)") {
        auto m2 = NdArray::Arange(3.f);
        auto m3 = NdArray::Arange(3.f) + 1.f;
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m2,
                "[[0, 2, 4],\n"
                " [3, 5, 7]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Add));
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m2,
                "[[0, 0, 0],\n"
                " [3, 3, 3]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Subtract));
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m2,
                "[[0, 1, 4],\n"
                " [0, 4, 10]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Multiply));
        CheckNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m3,
                "[[0, 0.5, 0.666667],\n"
                " [3, 2, 1.66667]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Divide));
    }

    SECTION("Function Arithmetic (NdArray, float) (inplace)") {
        CheckNdArrayInplace(NdArray::Arange(3.f), 2.f, "[2, 3, 4]",
                            static_cast<NdArray (*)(NdArray&&, float)>(Add));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 2.f, "[-2, -1, 0]",
                static_cast<NdArray (*)(NdArray&&, float)>(Subtract));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 2.f, "[0, 2, 4]",
                static_cast<NdArray (*)(NdArray&&, float)>(Multiply));
        CheckNdArrayInplace(NdArray::Arange(3.f), 2.f, "[0, 0.5, 1]",
                            static_cast<NdArray (*)(NdArray&&, float)>(Divide));
    }

    SECTION("Function Arithmetic (float, NdArray) (inplace)") {
        CheckNdArrayInplace(2.f, NdArray::Arange(3.f), "[2, 3, 4]",
                            static_cast<NdArray (*)(float, NdArray&&)>(Add));
        CheckNdArrayInplace(
                2.f, NdArray::Arange(3.f), "[2, 1, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(Subtract));
        CheckNdArrayInplace(
                2.f, NdArray::Arange(3.f), "[0, 2, 4]",
                static_cast<NdArray (*)(float, NdArray&&)>(Multiply));
        CheckNdArrayInplace(2.f, NdArray::Arange(3.f), "[inf, 2, 1]",
                            static_cast<NdArray (*)(float, NdArray&&)>(Divide));
    }

    SECTION("Function Comparison (NdArray, NdArray) (inplace both)") {
        CheckNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[0, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Equal));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[1, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(NotEqual));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[0, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Greater));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[0, 1, 1]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(GreaterEqual));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[1, 0, 0]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Less));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[1, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(LessEqual));
    }

    SECTION("Function Comparison (NdArray, NdArray) (inplace right)") {
        auto m2 = NdArray::Zeros(1) + 1.f;
        CheckNdArrayInplace(
                NdArray::Arange(3.f), m2, "[0, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Equal));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), m2, "[1, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(NotEqual));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), m2, "[0, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Greater));
        CheckNdArrayInplace(NdArray::Arange(3.f), m2, "[0, 1, 1]",
                            static_cast<NdArray (*)(NdArray&&, const NdArray&)>(
                                    GreaterEqual));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), m2, "[1, 0, 0]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Less));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), m2, "[1, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(LessEqual));
    }

    SECTION("Function Comparison (NdArray, NdArray) (inplace left)") {
        auto m1 = NdArray::Zeros(1) + 1.f;
        CheckNdArrayInplace(
                m1, NdArray::Arange(3.f), "[0, 1, 0]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Equal));
        CheckNdArrayInplace(
                m1, NdArray::Arange(3.f), "[1, 0, 1]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(NotEqual));
        CheckNdArrayInplace(
                m1, NdArray::Arange(3.f), "[1, 0, 0]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Greater));
        CheckNdArrayInplace(m1, NdArray::Arange(3.f), "[1, 1, 0]",
                            static_cast<NdArray (*)(const NdArray&, NdArray&&)>(
                                    GreaterEqual));
        CheckNdArrayInplace(
                m1, NdArray::Arange(3.f), "[0, 0, 1]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Less));
        CheckNdArrayInplace(
                m1, NdArray::Arange(3.f), "[0, 1, 1]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(LessEqual));
    }

    SECTION("Function Comparison (NdArray, float) (inplace)") {
        CheckNdArrayInplace(NdArray::Arange(3.f), 1.f, "[0, 1, 0]",
                            static_cast<NdArray (*)(NdArray&&, float)>(Equal));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[1, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(NotEqual));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[0, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(Greater));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[0, 1, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(GreaterEqual));
        CheckNdArrayInplace(NdArray::Arange(3.f), 1.f, "[1, 0, 0]",
                            static_cast<NdArray (*)(NdArray&&, float)>(Less));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[1, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, float)>(LessEqual));
    }

    SECTION("Function Comparison (NdArray, float) (inplace)") {
        CheckNdArrayInplace(1.f, NdArray::Arange(3.f), "[0, 1, 0]",
                            static_cast<NdArray (*)(float, NdArray&&)>(Equal));
        CheckNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[1, 0, 1]",
                static_cast<NdArray (*)(float, NdArray&&)>(NotEqual));
        CheckNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[1, 0, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(Greater));
        CheckNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[1, 1, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(GreaterEqual));
        CheckNdArrayInplace(1.f, NdArray::Arange(3.f), "[0, 0, 1]",
                            static_cast<NdArray (*)(float, NdArray&&)>(Less));
        CheckNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[0, 1, 1]",
                static_cast<NdArray (*)(float, NdArray&&)>(LessEqual));
    }

    SECTION("Function Basic Math (in-place)") {
        CheckNdArrayInplace(-NdArray::Arange(3.f), "[0, 1, 2]",
                            static_cast<NdArray (*)(NdArray &&)>(Abs));
        CheckNdArrayInplace(NdArray::Arange(7.f) / 3.f - 1.f,
                            "[-1, -1, -1, 0, 1, 1, 1]",
                            static_cast<NdArray (*)(NdArray &&)>(Sign));
        CheckNdArrayInplace(NdArray::Arange(7.f) / 3.f - 1.f,
                            "[-1, -0, -0, 0, 1, 1, 1]",
                            static_cast<NdArray (*)(NdArray &&)>(Ceil));
        CheckNdArrayInplace(NdArray::Arange(7.f) / 3.f - 1.f,
                            "[-1, -1, -1, 0, 0, 0, 1]",
                            static_cast<NdArray (*)(NdArray &&)>(Floor));
        auto clip_bind = std::bind(
                static_cast<NdArray (*)(NdArray&&, float, float)>(Clip),
                std::placeholders::_1, -0.5, 0.4);
        CheckNdArrayInplace(NdArray::Arange(7.f) / 3.f - 1.f,
                            "[-0.5, -0.5, -0.333333, 0, 0.333333, 0.4, 0.4]",
                            clip_bind);
        CheckNdArrayInplace(NdArray::Arange(3.f), "[0, 1, 1.41421]",
                            static_cast<NdArray (*)(NdArray &&)>(Sqrt));
        CheckNdArrayInplace(NdArray::Arange(3.f), "[1, 2.71828, 7.38906]",
                            static_cast<NdArray (*)(NdArray &&)>(Exp));
        CheckNdArrayInplace(NdArray::Arange(3.f), "[-inf, 0, 0.693147]",
                            static_cast<NdArray (*)(NdArray &&)>(Log));
        CheckNdArrayInplace(NdArray::Arange(3.f), "[0, 1, 4]",
                            static_cast<NdArray (*)(NdArray &&)>(Square));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Arange(3.f) + 1.f, "[0, 1, 8]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Power));
        const auto m1 = NdArray::Arange(3.f) + 1.f;
        const auto m2 = NdArray::Arange(3.f);
        CheckNdArrayInplace(
                m2, NdArray::Arange(3.f) + 1.f, "[0, 1, 8]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Power));
        CheckNdArrayInplace(
                NdArray::Arange(3.f), m1, "[0, 1, 8]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Power));
        CheckNdArrayInplace(NdArray::Arange(3.f), 4.f, "[0, 1, 16]",
                            static_cast<NdArray (*)(NdArray&&, float)>(Power));
        CheckNdArrayInplace(4.f, NdArray::Arange(3.f), "[1, 4, 16]",
                            static_cast<NdArray (*)(float, NdArray&&)>(Power));
    }

    SECTION("Function Trigonometric (in-place)") {
        CheckNdArrayInplace(NdArray::Arange(3.f), "[0, 0.841471, 0.909297]",
                            static_cast<NdArray (*)(NdArray &&)>(Sin));
        CheckNdArrayInplace(NdArray::Arange(3.f), "[1, 0.540302, -0.416147]",
                            static_cast<NdArray (*)(NdArray &&)>(Cos));
        CheckNdArrayInplace(NdArray::Arange(3.f), "[0, 1.55741, -2.18504]",
                            static_cast<NdArray (*)(NdArray &&)>(Tan));
    }

    SECTION("Function Inverse-Trigonometric (in-place)") {
        auto m1 = NdArray::Arange(3) - 1.f;
        auto m2 = NdArray::Arange(3) * 100.f;
        CheckNdArrayInplace(NdArray::Arange(3.f) - 1.f, "[-1.5708, 0, 1.5708]",
                            static_cast<NdArray (*)(NdArray &&)>(ArcSin));
        CheckNdArrayInplace(NdArray::Arange(3.f) - 1.f, "[3.14159, 1.5708, 0]",
                            static_cast<NdArray (*)(NdArray &&)>(ArcCos));
        CheckNdArrayInplace(NdArray::Arange(3.f) * 100.f, "[0, 1.5608, 1.5658]",
                            static_cast<NdArray (*)(NdArray &&)>(ArcTan));
        CheckNdArrayInplace(
                NdArray::Arange(3.f) - 1.f, NdArray::Arange(3.f) * 100.f,
                "[-1.5708, 0, 0.00499996]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(ArcTan2));
        CheckNdArrayInplace(
                m1, NdArray::Arange(3.f) * 100.f, "[-1.5708, 0, 0.00499996]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(ArcTan2));
        CheckNdArrayInplace(
                NdArray::Arange(3.f) - 1.f, m2, "[-1.5708, 0, 0.00499996]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(ArcTan2));
        CheckNdArrayInplace(
                NdArray::Arange(3.f) - 1.f, 2.f, "[-0.463648, 0, 0.463648]",
                static_cast<NdArray (*)(NdArray&&, float)>(ArcTan2));
        CheckNdArrayInplace(
                2.f, NdArray::Arange(3.f) - 1.f, "[2.03444, 1.5708, 1.10715]",
                static_cast<NdArray (*)(float, NdArray&&)>(ArcTan2));
    }

    SECTION("Where (in-place)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        CheckNdArray(Where(2.f < m1, NdArray::Ones(2, 1), NdArray::Arange(3)),
                     "[[0, 1, 2],\n"
                     " [1, 1, 1]]");
        CheckNdArray(Where(2.f < m1, 1.f, NdArray::Arange(3)),
                     "[[0, 1, 2],\n"
                     " [1, 1, 1]]");
        CheckNdArray(Where(2.f < m1, NdArray::Arange(3), 0.f),
                     "[[0, 0, 0],\n"
                     " [0, 1, 2]]");
        CheckNdArray(Where(2.f < m1, 1.f, 0.f),
                     "[[0, 0, 0],\n"
                     " [1, 1, 1]]");
    }

    SECTION("Function Inverse (in-place)") {
        auto m1 = NdArray::Arange(4).reshape(2, 2) + 1.f;
        auto m2 = Inv(m1);
        CheckNdArrayInplace(NdArray::Arange(4).reshape(2, 2) + 1.f,
                            "[[-2, 1],\n"
                            " [1.5, -0.5]]",
                            static_cast<NdArray (*)(NdArray &&)>(Inv),
                            4);  // Low precision
    }
}
