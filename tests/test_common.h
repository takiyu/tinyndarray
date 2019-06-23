#include "Catch2/single_include/catch2/catch.hpp"

#include "../tinyndarray.h"

#include <iomanip>

using namespace tinyndarray;

static void RequireNdArray(const NdArray& m, const std::string& str,
                           int precision = -1) {
    std::stringstream ss;
    if (0 < precision) {
        ss << std::setprecision(4);
    }
    ss << m;
    REQUIRE(ss.str() == str);
}

static void RequireNdArrayInplace(NdArray&& x, const std::string& str,
                                  std::function<NdArray(NdArray&&)> f,
                                  int precision = -1) {
    uintptr_t x_id = x.id();
    const NdArray& y = f(std::move(x));
    REQUIRE(y.id() == x_id);
    RequireNdArray(y, str, precision);
}

static void RequireNdArrayInplace(
        NdArray&& lhs, NdArray&& rhs, const std::string& str,
        std::function<NdArray(NdArray&&, NdArray&&)> f, int precision = -1) {
    uintptr_t l_id = lhs.id();
    uintptr_t r_id = rhs.id();
    const NdArray& ret = f(std::move(lhs), std::move(rhs));
    REQUIRE((ret.id() == l_id || ret.id() == r_id));
    RequireNdArray(ret, str, precision);
}

static void RequireNdArrayInplace(
        const NdArray& lhs, NdArray&& rhs, const std::string& str,
        std::function<NdArray(const NdArray&, NdArray&&)> f,
        int precision = -1) {
    uintptr_t l_id = lhs.id();
    uintptr_t r_id = rhs.id();
    const NdArray& ret = f(lhs, std::move(rhs));
    REQUIRE((ret.id() == l_id || ret.id() == r_id));
    RequireNdArray(ret, str, precision);
}

static void RequireNdArrayInplace(
        NdArray&& lhs, const NdArray& rhs, const std::string& str,
        std::function<NdArray(NdArray&&, const NdArray&)> f,
        int precision = -1) {
    uintptr_t l_id = lhs.id();
    uintptr_t r_id = rhs.id();
    const NdArray& ret = f(std::move(lhs), rhs);
    REQUIRE((ret.id() == l_id || ret.id() == r_id));
    RequireNdArray(ret, str, precision);
}

static void RequireNdArrayInplace(NdArray&& lhs, float rhs,
                                  const std::string& str,
                                  std::function<NdArray(NdArray&&, float)> f,
                                  int precision = -1) {
    uintptr_t l_id = lhs.id();
    const NdArray& ret = f(std::move(lhs), rhs);
    REQUIRE((ret.id() == l_id));
    RequireNdArray(ret, str, precision);
}

static void RequireNdArrayInplace(float lhs, NdArray&& rhs,
                                  const std::string& str,
                                  std::function<NdArray(float, NdArray&&)> f,
                                  int precision = -1) {
    uintptr_t r_id = rhs.id();
    const NdArray& ret = f(lhs, std::move(rhs));
    REQUIRE((ret.id() == r_id));
    RequireNdArray(ret, str, precision);
}

static void RequireNdArrayNotInplace(NdArray&& x, const std::string& str,
                                     std::function<NdArray(const NdArray&)> f,
                                     int precision = -1) {
    uintptr_t x_id = x.id();
    const NdArray& y = f(x);
    REQUIRE(y.id() != x_id);
    RequireNdArray(y, str, precision);
}

static bool IsSameNdArray(const NdArray& m1, const NdArray& m2) {
    if (m1.shape() != m2.shape()) {
        return false;
    }
    auto data1 = m1.data();
    auto data2 = m2.data();
    for (size_t i = 0; i < m1.size(); i++) {
        if (*(data1++) != *(data2++)) {
            return false;
        }
    }
    return true;
}

TEST_CASE("NdArray") {
    // -------------------------- Basic construction ---------------------------
    SECTION("Empty") {
        const NdArray m1;
        REQUIRE(m1.empty());
        REQUIRE(m1.size() == 0);
        REQUIRE(m1.shape() == Shape{0});
        REQUIRE(m1.ndim() == 1);
    }

    // --------------------------- Float initializer ---------------------------
    SECTION("Float initializer") {
        const NdArray m1 = {1.f, 2.f, 3.f};
        const NdArray m2 = {{1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
        const NdArray m3 = {{{1.f, 2.f}}, {{3.f, 4.f}}, {{2.f, 3.f}}};
        REQUIRE(m1.shape() == Shape{3});
        REQUIRE(m2.shape() == Shape{2, 3});
        REQUIRE(m3.shape() == Shape{3, 1, 2});
        RequireNdArray(m1, "[1, 2, 3]");
        RequireNdArray(m2,
                       "[[1, 2, 3],\n"
                       " [4, 5, 6]]");
        RequireNdArray(m3,
                       "[[[1, 2]],\n"
                       " [[3, 4]],\n"
                       " [[2, 3]]]");
    }

    SECTION("Float initializer invalid") {
        REQUIRE_NOTHROW(NdArray{{{1.f, 2.f}}, {{3.f, 4.f}}, {{1.f, 2.f}}});
        REQUIRE_THROWS(NdArray{{{1, 2}}, {}});
        REQUIRE_THROWS(NdArray{{{1.f, 2.f}}, {{3.f, 4.f}}, {{1.f, 2.f, 3.f}}});
    }

    SECTION("Confusable initializers") {
        const NdArray m1 = {1.f, 2.f, 3.f};  // Float initializer
        const NdArray m2 = {1, 2, 3};        // Shape (int) initalizer
        const NdArray m3 = {{1, 2, 3}};      // Float initializer due to nest
        REQUIRE(m1.shape() == Shape{3});
        REQUIRE(m2.shape() == Shape{1, 2, 3});
        REQUIRE(m3.shape() == Shape{1, 3});
    }

    // --------------------------- Static initializer --------------------------
    SECTION("Empty/Ones/Zeros") {
        const NdArray m1({2, 5});  // Same as Empty
        const auto m2 = NdArray::Empty({2, 5});
        const auto m3 = NdArray::Zeros({2, 5});
        const auto m4 = NdArray::Ones({2, 5});
        REQUIRE(m1.shape() == Shape{2, 5});
        REQUIRE(m2.shape() == Shape{2, 5});
        REQUIRE(m3.shape() == Shape{2, 5});
        REQUIRE(m4.shape() == Shape{2, 5});
        RequireNdArray(m3,
                       "[[0, 0, 0, 0, 0],\n"
                       " [0, 0, 0, 0, 0]]");
        RequireNdArray(m4,
                       "[[1, 1, 1, 1, 1],\n"
                       " [1, 1, 1, 1, 1]]");
    }

    SECTION("Empty/Ones/Zeros by template") {
        const NdArray m1({2, 5});  // No template support
        const auto m2 = NdArray::Empty(2, 5);
        const auto m3 = NdArray::Zeros(2, 5);
        const auto m4 = NdArray::Ones(2, 5);
        REQUIRE(m1.shape() == Shape{2, 5});
        REQUIRE(m2.shape() == Shape{2, 5});
        REQUIRE(m3.shape() == Shape{2, 5});
        REQUIRE(m4.shape() == Shape{2, 5});
        RequireNdArray(m3,
                       "[[0, 0, 0, 0, 0],\n"
                       " [0, 0, 0, 0, 0]]");
        RequireNdArray(m4,
                       "[[1, 1, 1, 1, 1],\n"
                       " [1, 1, 1, 1, 1]]");
    }

    SECTION("Arange") {
        const auto m1 = NdArray::Arange(10.f);
        const auto m2 = NdArray::Arange(0.f, 10.f, 1.f);
        const auto m3 = NdArray::Arange(5.f, 5.5f, 0.1f);
        REQUIRE(m1.shape() == Shape{10});
        REQUIRE(m2.shape() == Shape{10});
        REQUIRE(m3.shape() == Shape{5});
        RequireNdArray(m1, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
        RequireNdArray(m2, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]");
        RequireNdArray(m3, "[5, 5.1, 5.2, 5.3, 5.4]");
    }

    // --------------------------------- Random --------------------------------
    SECTION("Random uniform") {
        NdArray::Seed(0);
        const auto m1 = NdArray::Uniform({2, 3});
        NdArray::Seed(0);
        const auto m2 = NdArray::Uniform({2, 3});
        NdArray::Seed(1);
        const auto m3 = NdArray::Uniform({2, 3});
        REQUIRE(IsSameNdArray(m1, m2));
        REQUIRE(!IsSameNdArray(m1, m3));
    }

    SECTION("Random normal") {
        NdArray::Seed(0);
        const auto m1 = NdArray::Normal({2, 3});
        NdArray::Seed(0);
        const auto m2 = NdArray::Normal({2, 3});
        NdArray::Seed(1);
        const auto m3 = NdArray::Normal({2, 3});
        REQUIRE(IsSameNdArray(m1, m2));
        REQUIRE(!IsSameNdArray(m1, m3));
    }

    // ------------------------------ Basic method -----------------------------
    SECTION("Basic method") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = m1.copy();
        m2.fill(-1);
        RequireNdArray(m1,
                       "[[0, 1, 2],\n"
                       " [3, 4, 5]]");
        RequireNdArray(m2,
                       "[[-1, -1, -1],\n"
                       " [-1, -1, -1]]");
        REQUIRE(m1.ndim() == 2);
        REQUIRE(m1.flatten().ndim() == 1);
        REQUIRE(m1.flatten().id() != m1.id());  // Copy
        REQUIRE(m1.ravel().ndim() == 1);
        REQUIRE(m1.ravel().id() == m1.id());  // Same instance
    }

    // ------------------------------- Begin/End -------------------------------
    SECTION("Begin/End") {
        auto m1 = NdArray::Arange(1.f, 10.01f);
        // C++11 for-loop
        float sum1 = 0.f;
        for (auto&& v : m1) {
            sum1 += v;
        }
        REQUIRE(sum1 == Approx(55.f));
        // std library
        float sum2 = std::accumulate(m1.begin(), m1.end(), 0.f);
        REQUIRE(sum2 == Approx(55.f));
    }

    // ------------------------------- Float cast ------------------------------
    SECTION("Float cast") {
        auto m1 = NdArray::Ones({1, 1});
        auto m2 = NdArray::Ones({1, 2});
        REQUIRE(static_cast<float>(m1) == 1);
        REQUIRE_THROWS(static_cast<float>(m2));
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
        RequireNdArray(m1, "[0, 1, 2, -1, 4, 5, 6, 7, 8, 9, -2, 11]");
        RequireNdArray(m2,
                       "[[0, 1, 2, 3],\n"
                       " [4, -1, 6, 7],\n"
                       " [8, 9, 10, -2]]");
        RequireNdArray(m3,
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
        RequireNdArray(m1, "[0, 1, 2, -1, 4, 5, 6, 7, 8, 9, -2, 11]");
        RequireNdArray(m2,
                       "[[0, 1, 2, 3],\n"
                       " [4, -1, 6, 7],\n"
                       " [8, 9, 10, -2]]");
        RequireNdArray(m3,
                       "[[[0, 1, 2],\n"
                       "  [3, -2, 5]],\n"
                       " [[6, 7, 8],\n"
                       "  [9, 10, -1]]]");
    }

    SECTION("Index access by [] (const)") {
        const auto m1 = NdArray::Arange(12.f).reshape({1, 4, 3});
        REQUIRE(m1[{0, 2, 1}] == 7);
        REQUIRE(m1[{0, -1, 1}] == 10);
    }

    SECTION("Index access by () (const)") {
        const auto m1 = NdArray::Arange(12.f).reshape({1, 4, 3});
        REQUIRE(m1(0, 2, 1) == 7);
        REQUIRE(m1(0, -1, 1) == 10);
    }

    // -------------------------------- Reshape --------------------------------
    SECTION("Reshape") {
        auto m1 = NdArray::Arange(12.f);
        auto m2 = m1.reshape({3, 4});
        auto m3 = m2.reshape({2, -1});
        auto m4 = m3.reshape({2, 2, -1});
        RequireNdArray(m1, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]");
        RequireNdArray(m2,
                       "[[0, 1, 2, 3],\n"
                       " [4, 5, 6, 7],\n"
                       " [8, 9, 10, 11]]");
        RequireNdArray(m3,
                       "[[0, 1, 2, 3, 4, 5],\n"
                       " [6, 7, 8, 9, 10, 11]]");
        RequireNdArray(m4,
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
        RequireNdArray(m1, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]");
        RequireNdArray(m2,
                       "[[0, 1, 2, 3],\n"
                       " [4, 5, 6, 7],\n"
                       " [8, 9, 10, 11]]");
        RequireNdArray(m3,
                       "[[0, 1, 2, 3, 4, 5],\n"
                       " [6, 7, 8, 9, 10, 11]]");
        RequireNdArray(m4,
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
        RequireNdArray(m1, "[-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]");
        RequireNdArray(m2,
                       "[[-1, 1, 2, 3],\n"
                       " [4, 5, 6, 7],\n"
                       " [8, 9, 10, 11]]");
        RequireNdArray(m3,
                       "[[-1, 1, 2, 3, 4, 5],\n"
                       " [6, 7, 8, 9, 10, 11]]");
        RequireNdArray(m4,
                       "[[[-1, 1, 2],\n"
                       "  [3, 4, 5]],\n"
                       " [[6, 7, 8],\n"
                       "  [9, 10, 11]]]");
    }

    SECTION("Reshape invalid") {
        auto m1 = NdArray::Arange(12.f);
        REQUIRE_THROWS(m1.reshape({5, 2}));
        REQUIRE_THROWS(m1.reshape({-1, -1}));
    }

    SECTION("Flatten") {
        auto m1 = NdArray::Arange(12.f).reshape(2, 2, 3);
        auto m2 = m1.flatten();
        RequireNdArray(m2, "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]");
    }

    // --------------------------------- Slice ---------------------------------
    SECTION("Slice 2-dim") {
        auto m1 = NdArray::Arange(16.f).reshape(4, 4);
        auto m2 = m1.slice({{1, 3}, {1, 3}});
        auto m3 = m1.slice({1, 3}, {0, 4});
        auto m4 = m1.slice({0, 4}, {1, 3});
        auto m5 = m1.slice({1, -1}, {0, 100000});
        REQUIRE(m1.shape() == Shape{4, 4});
        REQUIRE(m2.shape() == Shape{2, 2});
        REQUIRE(m3.shape() == Shape{2, 4});
        REQUIRE(m4.shape() == Shape{4, 2});
        REQUIRE(m5.shape() == Shape{2, 4});
        RequireNdArray(m1,
                       "[[0, 1, 2, 3],\n"
                       " [4, 5, 6, 7],\n"
                       " [8, 9, 10, 11],\n"
                       " [12, 13, 14, 15]]");
        RequireNdArray(m2,
                       "[[5, 6],\n"
                       " [9, 10]]");
        RequireNdArray(m3,
                       "[[4, 5, 6, 7],\n"
                       " [8, 9, 10, 11]]");
        RequireNdArray(m4,
                       "[[1, 2],\n"
                       " [5, 6],\n"
                       " [9, 10],\n"
                       " [13, 14]]");
        RequireNdArray(m5,
                       "[[4, 5, 6, 7],\n"
                       " [8, 9, 10, 11]]");
    }

    SECTION("Slice high-dim") {
        auto m1 = NdArray::Arange(256.f).reshape(4, 4, 4, 4);
        auto m2 = m1.slice({{1, 3}, {1, 3}, {1, 3}, {1, 3}});
        auto m3 = m1.slice({1, 3}, {1, 3}, {1, 3}, {1, 3});
        REQUIRE(m1.shape() == Shape{4, 4, 4, 4});
        REQUIRE(m2.shape() == Shape{2, 2, 2, 2});
        REQUIRE(m3.shape() == Shape{2, 2, 2, 2});
        RequireNdArray(m2,
                       "[[[[85, 86],\n"
                       "   [89, 90]],\n"
                       "  [[101, 102],\n"
                       "   [105, 106]]],\n"
                       " [[[149, 150],\n"
                       "   [153, 154]],\n"
                       "  [[165, 166],\n"
                       "   [169, 170]]]]");
        RequireNdArray(m3,
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
        REQUIRE_THROWS(m1.dot(m1));
    }

    SECTION("Dot (scalar)") {
        // Scalar multiply
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m1_a1 = m1.dot(2.f);
        auto m1_a2 = m1.dot(NdArray{2.f});
        auto m1_b = NdArray({2.f}).dot(m1);
        RequireNdArray(m1_a1,
                       "[[0, 2, 4],\n"
                       " [6, 8, 10]]");
        RequireNdArray(m1_a2,
                       "[[0, 2, 4],\n"
                       " [6, 8, 10]]");
        RequireNdArray(m1_b,
                       "[[0, 2, 4],\n"
                       " [6, 8, 10]]");
    }

    SECTION("Dot (1D, 1D)") {
        // Inner product of vectors
        auto m1 = NdArray::Arange(3.f);
        auto m2 = NdArray::Ones(3);
        float m11 = m1.dot(m1);
        float m12 = m1.dot(m2);
        REQUIRE(m11 == Approx(5.f));
        REQUIRE(m12 == Approx(3.f));
        // Shape mismatch
        auto m3 = NdArray::Arange(4.f);
        REQUIRE_THROWS(m1.dot(m3));
    }

    SECTION("Dot (2D, 2D)") {
        // Inner product of 2D matrix
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(6.f).reshape(3, 2);
        auto m12 = m1.dot(m2);
        RequireNdArray(m12,
                       "[[10, 13],\n"
                       " [28, 40]]");
        // Shape mismatch
        REQUIRE_THROWS(m1.dot(m1));
    }

    SECTION("Dot (2D, 1D)") {
        // Inner product of 2D matrix and vector (2D, 1D)
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(3.f);
        auto m12 = m1.dot(m2);
        RequireNdArray(m12, "[5, 14]");
        // Shape mismatch
        auto m3 = NdArray::Arange(2.f);
        REQUIRE_THROWS(m1.dot(m3));
    }

    SECTION("Dot (ND, 1D)") {
        // Inner product of ND matrix and vector (ND, 1D)
        auto m1 = NdArray::Arange(12.f).reshape(2, 2, 3);
        auto m2 = NdArray::Arange(3.f);
        auto m12 = m1.dot(m2);
        RequireNdArray(m12,
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
        REQUIRE(m12.shape() == Shape{2, 3, 3});
        REQUIRE(m13.shape() == Shape{2, 3, 3, 2});
        RequireNdArray(m12,
                       "[[[3, 4, 5],\n"
                       "  [9, 14, 19],\n"
                       "  [15, 24, 33]],\n"
                       " [[21, 34, 47],\n"
                       "  [27, 44, 61],\n"
                       "  [33, 54, 75]]]");
        RequireNdArray(m13,
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

    // ----------------------------- Cross product -----------------------------
    SECTION("Cross (1D, 1D), (3, 3 elem)") {
        NdArray m1 = {1.f, 2.f, 3.f};
        NdArray m2 = {4.f, 5.f, 6.f};
        auto m12 = m1.cross(m2);
        RequireNdArray(m12, "[-3, 6, -3]");
    }

    SECTION("Cross (1D, 1D), (3, 2 elem)") {
        NdArray m1 = {1.f, 2.f, 3.f};
        NdArray m2 = {4.f, 5.f};
        auto m12 = m1.cross(m2);
        RequireNdArray(m12, "[-15, 12, -3]");
    }

    SECTION("Cross (1D, 1D), (2, 3 elem)") {
        NdArray m1 = {1.f, 2.f};
        NdArray m2 = {4.f, 5.f, 6.f};
        auto m12 = m1.cross(m2);
        RequireNdArray(m12, "[12, -6, -3]");
    }

    SECTION("Cross (1D, 1D), (2, 2 elem)") {
        NdArray m1 = {1.f, 2.f};
        NdArray m2 = {4.f, 5.f};
        float m12 = m1.cross(m2);
        REQUIRE(m12 == Approx(-3.f));
    }

    SECTION("Cross (1D, 1D), (mismatch)") {
        NdArray m1 = {1.f};
        NdArray m2 = {4.f, 5.f};
        NdArray m3 = {4.f, 5.f, 6.f, 7.f};
        REQUIRE_THROWS(m1.cross(m2));
        REQUIRE_THROWS(m2.cross(m3));
    }

    SECTION("Cross (ND, MD), (3, 3 elem)") {
        auto m1 = NdArray::Arange(18.f).reshape(3, 2, 3);
        auto m2 = NdArray::Arange(6.f).reshape(2, 3) + 1.f;
        auto m12 = m1.cross(m2);
        RequireNdArray(m12,
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
        RequireNdArray(m12,
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
        RequireNdArray(m12,
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
        RequireNdArray(m12,
                       "[[-1, -1],\n"
                       " [3, 3],\n"
                       " [7, 7]]");
    }

    // ----------------------------- Axis operation ----------------------------
    SECTION("Sum") {
        NdArray m0;
        auto m1 = NdArray::Arange(6.f);
        auto m2 = NdArray::Arange(36.f).reshape(2, 3, 2, 3);
        RequireNdArray(m0.sum(), "[0]");
        RequireNdArray(m1.sum(), "[15]");
        RequireNdArray(m2.sum({0}),
                       "[[[18, 20, 22],\n"
                       "  [24, 26, 28]],\n"
                       " [[30, 32, 34],\n"
                       "  [36, 38, 40]],\n"
                       " [[42, 44, 46],\n"
                       "  [48, 50, 52]]]");
        RequireNdArray(m2.sum({2}),
                       "[[[3, 5, 7],\n"
                       "  [15, 17, 19],\n"
                       "  [27, 29, 31]],\n"
                       " [[39, 41, 43],\n"
                       "  [51, 53, 55],\n"
                       "  [63, 65, 67]]]");
        RequireNdArray(m2.sum({3}),
                       "[[[3, 12],\n"
                       "  [21, 30],\n"
                       "  [39, 48]],\n"
                       " [[57, 66],\n"
                       "  [75, 84],\n"
                       "  [93, 102]]]");
        RequireNdArray(m2.sum({1, 2}),
                       "[[45, 51, 57],\n"
                       " [153, 159, 165]]");
        RequireNdArray(m2.sum({1, 3}),
                       "[[63, 90],\n"
                       " [225, 252]]");
    }

    SECTION("Min") {
        NdArray m0;
        auto m1 = NdArray::Arange(6.f) - 3.f;
        auto m2 = NdArray::Arange(12.f).reshape(2, 3, 2) - 6.f;
        REQUIRE_THROWS(m0.min());
        RequireNdArray(m1.min(), "[-3]");
        RequireNdArray(m2.min({0}),
                       "[[-6, -5],\n"
                       " [-4, -3],\n"
                       " [-2, -1]]");
        RequireNdArray(m2.min({2}),
                       "[[-6, -4, -2],\n"
                       " [0, 2, 4]]");
        RequireNdArray(m2.min({2, 1}), "[-6, 0]");
    }

    SECTION("Max") {
        NdArray m0;
        auto m1 = NdArray::Arange(6.f) - 3.f;
        auto m2 = NdArray::Arange(12.f).reshape(2, 3, 2) - 6.f;
        REQUIRE_THROWS(m0.max());
        RequireNdArray(m1.max(), "[2]");
        RequireNdArray(m2.max({0}),
                       "[[0, 1],\n"
                       " [2, 3],\n"
                       " [4, 5]]");
        RequireNdArray(m2.max({2}),
                       "[[-5, -3, -1],\n"
                       " [1, 3, 5]]");
        RequireNdArray(m2.max({2, 1}), "[-1, 5]");
    }

    SECTION("Mean") {
        NdArray m0;
        auto m1 = NdArray::Arange(6.f) - 3.f;
        auto m2 = NdArray::Arange(12.f).reshape(2, 3, 2) - 6.f;
        RequireNdArray(m0.mean(), "[nan]");
        RequireNdArray(m1.mean(), "[-0.5]");
        RequireNdArray(m2.mean({0}),
                       "[[-3, -2],\n"
                       " [-1, 0],\n"
                       " [1, 2]]");
        RequireNdArray(m2.mean({2}),
                       "[[-5.5, -3.5, -1.5],\n"
                       " [0.5, 2.5, 4.5]]");
        RequireNdArray(m2.mean({2, 1}), "[-3.5, 2.5]");
    }

    // ------------------------------- Operator --------------------------------
    SECTION("Single +- operators") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m_p = +m1;
        auto m_n = -m1;
        REQUIRE(m_p.id() != m1.id());
        REQUIRE(m_n.id() != m1.id());
        RequireNdArray(m_p,
                       "[[0, 1, 2],\n"
                       " [3, 4, 5]]");
        RequireNdArray(m_n,
                       "[[-0, -1, -2],\n"
                       " [-3, -4, -5]]");
    }

    SECTION("Add same shape") {
        auto m1 = NdArray::Arange(12.f).reshape(2, 3, 2);
        auto m2 = NdArray::Ones({2, 3, 2});
        auto m3 = m1 + m2;
        REQUIRE(m1.shape() == m2.shape());
        REQUIRE(m1.shape() == m3.shape());
        RequireNdArray(m3,
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
        REQUIRE(m12.shape() == Shape{2, 3});
        REQUIRE(m13.shape() == Shape{2, 3});
        REQUIRE(m23.shape() == Shape{2, 3});
        RequireNdArray(m12,
                       "[[0, 1, 2],\n"
                       " [4, 5, 6]]");
        RequireNdArray(m13,
                       "[[0, 2, 4],\n"
                       " [3, 5, 7]]");
        RequireNdArray(m23,
                       "[[0, 1, 2],\n"
                       " [1, 2, 3]]");
    }

    SECTION("Add broadcast high-dim") {
        auto m1 = NdArray::Arange(6.f).reshape(1, 2, 1, 1, 3);
        auto m2 = NdArray::Arange(2.f).reshape(2, 1);
        auto m3 = NdArray::Arange(3.f).reshape(1, 3);
        auto m12 = m1 + m2;
        auto m13 = m1 + m3;
        REQUIRE(m12.shape() == Shape{1, 2, 1, 2, 3});
        REQUIRE(m13.shape() == Shape{1, 2, 1, 1, 3});
        RequireNdArray(m12,
                       "[[[[[0, 1, 2],\n"
                       "    [1, 2, 3]]],\n"
                       "  [[[3, 4, 5],\n"
                       "    [4, 5, 6]]]]]");
        RequireNdArray(m13,
                       "[[[[[0, 2, 4]]],\n"
                       "  [[[3, 5, 7]]]]]");
    }

    SECTION("Add empty") {
        NdArray m1;
        auto m2 = m1 + 1.f;
        REQUIRE(m1.shape() == Shape{0});
        REQUIRE(m2.shape() == Shape{0});
        REQUIRE(m2.size() == 0);
    }

    SECTION("Sub/Mul/Div") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(3.f).reshape(3);
        auto m_sub = m1 - m2;
        auto m_mul = m1 * m2;
        auto m_div = m1 / m2;
        REQUIRE(m_sub.shape() == Shape{2, 3});
        REQUIRE(m_mul.shape() == Shape{2, 3});
        REQUIRE(m_div.shape() == Shape{2, 3});
        RequireNdArray(m_sub,
                       "[[0, 0, 0],\n"
                       " [3, 3, 3]]");
        RequireNdArray(m_mul,
                       "[[0, 1, 4],\n"
                       " [0, 4, 10]]");
        // `0.f / 0.f` can be both of `nan` and `-nan`.
        m_div(0, 0) = std::abs(m_div(0, 0));
        RequireNdArray(m_div,
                       "[[nan, 1, 1],\n"
                       " [inf, 4, 2.5]]");
    }

    SECTION("Arithmetic operators (NdArray, float)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m_add = m1 + 10.f;
        auto m_sub = m1 - 10.f;
        auto m_mul = m1 * 10.f;
        auto m_div = m1 / 10.f;
        RequireNdArray(m_add,
                       "[[10, 11, 12],\n"
                       " [13, 14, 15]]");
        RequireNdArray(m_sub,
                       "[[-10, -9, -8],\n"
                       " [-7, -6, -5]]");
        RequireNdArray(m_mul,
                       "[[0, 10, 20],\n"
                       " [30, 40, 50]]");
        RequireNdArray(m_div,
                       "[[0, 0.1, 0.2],\n"
                       " [0.3, 0.4, 0.5]]");
    }

    SECTION("Arithmetic operators (float, NdArray)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m_add = 10.f + m1;
        auto m_sub = 10.f - m1;
        auto m_mul = 10.f * m1;
        auto m_div = 10.f / m1;
        RequireNdArray(m_add,
                       "[[10, 11, 12],\n"
                       " [13, 14, 15]]");
        RequireNdArray(m_sub,
                       "[[10, 9, 8],\n"
                       " [7, 6, 5]]");
        RequireNdArray(m_mul,
                       "[[0, 10, 20],\n"
                       " [30, 40, 50]]");
        RequireNdArray(m_div,
                       "[[inf, 10, 5],\n"
                       " [3.33333, 2.5, 2]]");
    }

    SECTION("Comparison operators (NdArray, NdArray)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(2.f).reshape(2, 1) + 3.f;
        RequireNdArray(m1 == m2,
                       "[[0, 0, 0],\n"
                       " [0, 1, 0]]");
        RequireNdArray(m1 != m2,
                       "[[1, 1, 1],\n"
                       " [1, 0, 1]]");
        RequireNdArray(m1 > m2,
                       "[[0, 0, 0],\n"
                       " [0, 0, 1]]");
        RequireNdArray(m1 >= m2,
                       "[[0, 0, 0],\n"
                       " [0, 1, 1]]");
        RequireNdArray(m1 < m2,
                       "[[1, 1, 1],\n"
                       " [1, 0, 0]]");
        RequireNdArray(m1 <= m2,
                       "[[1, 1, 1],\n"
                       " [1, 1, 0]]");
    }

    SECTION("Comparison operators (NdArray, float)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        RequireNdArray(m1 == 1.f,
                       "[[0, 1, 0],\n"
                       " [0, 0, 0]]");
        RequireNdArray(m1 != 1.f,
                       "[[1, 0, 1],\n"
                       " [1, 1, 1]]");
        RequireNdArray(m1 > 1.f,
                       "[[0, 0, 1],\n"
                       " [1, 1, 1]]");
        RequireNdArray(m1 >= 1.f,
                       "[[0, 1, 1],\n"
                       " [1, 1, 1]]");
        RequireNdArray(m1 < 1.f,
                       "[[1, 0, 0],\n"
                       " [0, 0, 0]]");
        RequireNdArray(m1 <= 1.f,
                       "[[1, 1, 0],\n"
                       " [0, 0, 0]]");
    }

    SECTION("Comparison operators (float, NdArray)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        RequireNdArray(1.f == m1,
                       "[[0, 1, 0],\n"
                       " [0, 0, 0]]");
        RequireNdArray(1.f != m1,
                       "[[1, 0, 1],\n"
                       " [1, 1, 1]]");
        RequireNdArray(1.f > m1,
                       "[[1, 0, 0],\n"
                       " [0, 0, 0]]");
        RequireNdArray(1.f >= m1,
                       "[[1, 1, 0],\n"
                       " [0, 0, 0]]");
        RequireNdArray(1.f < m1,
                       "[[0, 0, 1],\n"
                       " [1, 1, 1]]");
        RequireNdArray(1.f <= m1,
                       "[[0, 1, 1],\n"
                       " [1, 1, 1]]");
    }

    // --------------------------- In-place Operator ---------------------------
    SECTION("Arithmetic operators (NdArray, NdArray) (in-place both)") {
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f),
                "[[0, 2, 4],\n"
                " [3, 5, 7]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator+));
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f),
                "[[0, 0, 0],\n"
                " [3, 3, 3]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator-));
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f),
                "[[0, 1, 4],\n"
                " [0, 4, 10]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator*));
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f) + 1.f,
                "[[0, 0.5, 0.666667],\n"
                " [3, 2, 1.66667]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator/));
    }

    SECTION("Arithmetic operators (NdArray, NdArray) (in-place right)") {
        auto m1 = NdArray::Arange(3.f);
        auto m2 = NdArray::Arange(3.f) + 1.f;
        RequireNdArrayInplace(
                m1, NdArray::Arange(6.f).reshape(2, 3),
                "[[0, 2, 4],\n"
                " [3, 5, 7]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(operator+));
        RequireNdArrayInplace(
                m1, NdArray::Arange(6.f).reshape(2, 3),
                "[[0, 0, 0],\n"
                " [-3, -3, -3]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(operator-));
        RequireNdArrayInplace(
                m1, NdArray::Arange(6.f).reshape(2, 3),
                "[[0, 1, 4],\n"
                " [0, 4, 10]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(operator*));
        RequireNdArrayInplace(
                m2, NdArray::Arange(6.f).reshape(2, 3),
                "[[inf, 2, 1.5],\n"
                " [0.333333, 0.5, 0.6]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(operator/));
    }

    SECTION("Arithmetic operators (NdArray, NdArray) (in-place left)") {
        auto m2 = NdArray::Arange(3.f);
        auto m3 = NdArray::Arange(3.f) + 1.f;
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m2,
                "[[0, 2, 4],\n"
                " [3, 5, 7]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(operator+));
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m2,
                "[[0, 0, 0],\n"
                " [3, 3, 3]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(operator-));
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m2,
                "[[0, 1, 4],\n"
                " [0, 4, 10]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(operator*));
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m3,
                "[[0, 0.5, 0.666667],\n"
                " [3, 2, 1.66667]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(operator/));
    }

    SECTION("Arithmetic operators (NdArray, float) (inplace)") {
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 2.f, "[2, 3, 4]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator+));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 2.f, "[-2, -1, 0]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator-));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 2.f, "[0, 2, 4]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator*));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 2.f, "[0, 0.5, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator/));
    }

    SECTION("Arithmetic operators (float, NdArray) (inplace)") {
        RequireNdArrayInplace(
                2.f, NdArray::Arange(3.f), "[2, 3, 4]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator+));
        RequireNdArrayInplace(
                2.f, NdArray::Arange(3.f), "[2, 1, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator-));
        RequireNdArrayInplace(
                2.f, NdArray::Arange(3.f), "[0, 2, 4]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator*));
        RequireNdArrayInplace(
                2.f, NdArray::Arange(3.f), "[inf, 2, 1]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator/));
    }

    SECTION("Comparison operators (NdArray, NdArray) (inplace both)") {
        RequireNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[0, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator==));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[1, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator!=));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[0, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator>));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[0, 1, 1]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator>=));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[1, 0, 0]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator<));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[1, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(operator<=));
    }

    SECTION("Comparison operators (NdArray, NdArray) (inplace right)") {
        auto m2 = NdArray::Zeros(1) + 1.f;
        RequireNdArrayInplace(NdArray::Arange(3.f), m2, "[0, 1, 0]",
                              static_cast<NdArray (*)(
                                      NdArray&&, const NdArray&)>(operator==));
        RequireNdArrayInplace(NdArray::Arange(3.f), m2, "[1, 0, 1]",
                              static_cast<NdArray (*)(
                                      NdArray&&, const NdArray&)>(operator!=));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), m2, "[0, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(operator>));
        RequireNdArrayInplace(NdArray::Arange(3.f), m2, "[0, 1, 1]",
                              static_cast<NdArray (*)(
                                      NdArray&&, const NdArray&)>(operator>=));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), m2, "[1, 0, 0]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(operator<));
        RequireNdArrayInplace(NdArray::Arange(3.f), m2, "[1, 1, 0]",
                              static_cast<NdArray (*)(
                                      NdArray&&, const NdArray&)>(operator<=));
    }

    SECTION("Comparison operators (NdArray, NdArray) (inplace left)") {
        auto m1 = NdArray::Zeros(1) + 1.f;
        RequireNdArrayInplace(m1, NdArray::Arange(3.f), "[0, 1, 0]",
                              static_cast<NdArray (*)(const NdArray&,
                                                      NdArray&&)>(operator==));
        RequireNdArrayInplace(m1, NdArray::Arange(3.f), "[1, 0, 1]",
                              static_cast<NdArray (*)(const NdArray&,
                                                      NdArray&&)>(operator!=));
        RequireNdArrayInplace(
                m1, NdArray::Arange(3.f), "[1, 0, 0]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(operator>));
        RequireNdArrayInplace(m1, NdArray::Arange(3.f), "[1, 1, 0]",
                              static_cast<NdArray (*)(const NdArray&,
                                                      NdArray&&)>(operator>=));
        RequireNdArrayInplace(
                m1, NdArray::Arange(3.f), "[0, 0, 1]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(operator<));
        RequireNdArrayInplace(m1, NdArray::Arange(3.f), "[0, 1, 1]",
                              static_cast<NdArray (*)(const NdArray&,
                                                      NdArray&&)>(operator<=));
    }

    SECTION("Comparison operators (NdArray, float) (inplace)") {
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[0, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator==));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[1, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator!=));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[0, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator>));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[0, 1, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator>=));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[1, 0, 0]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator<));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[1, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, float)>(operator<=));
    }

    SECTION("Comparison operators (NdArray, float) (inplace)") {
        RequireNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[0, 1, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator==));
        RequireNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[1, 0, 1]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator!=));
        RequireNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[1, 0, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator>));
        RequireNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[1, 1, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator>=));
        RequireNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[0, 0, 1]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator<));
        RequireNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[0, 1, 1]",
                static_cast<NdArray (*)(float, NdArray&&)>(operator<=));
    }

    SECTION("Compound assignment operators (NdArray, NdArray) (in-plafce)") {
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
        RequireNdArray(m0,
                       "[[0, 2, 4],\n"
                       " [6, 8, 10]]");
        RequireNdArray(m1,
                       "[[0, 2, 4],\n"
                       " [3, 5, 7]]");
        RequireNdArray(m2,
                       "[[0, 0, 0],\n"
                       " [3, 3, 3]]");
        RequireNdArray(m3,
                       "[[0, 1, 4],\n"
                       " [0, 4, 10]]");
        // `0.f / 0.f` can be both of `nan` and `-nan`.
        m4(0, 0) = std::abs(m4(0, 0));
        RequireNdArray(m4,
                       "[[nan, 1, 1],\n"
                       " [inf, 4, 2.5]]");
        REQUIRE(m0.id() == m0_id);  // in-place
        REQUIRE(m1.id() == m1_id);
        REQUIRE(m2.id() == m2_id);
        REQUIRE(m3.id() == m3_id);
        REQUIRE(m4.id() == m4_id);
        // size change is not allowed
        REQUIRE_THROWS(m5 += m0);
        auto m6 = m0.reshape(2, 1, 3);
        REQUIRE_THROWS(m6 *= m5.reshape(3, 1));
    }

    SECTION("Compound assignment operators (NdArray, float) (in-plafce)") {
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
        RequireNdArray(m1,
                       "[[10, 11, 12],\n"
                       " [13, 14, 15]]");
        RequireNdArray(m2,
                       "[[-10, -9, -8],\n"
                       " [-7, -6, -5]]");
        RequireNdArray(m3,
                       "[[0, 10, 20],\n"
                       " [30, 40, 50]]");
        RequireNdArray(m4,
                       "[[0, 0.1, 0.2],\n"
                       " [0.3, 0.4, 0.5]]");
        REQUIRE(m1.id() == m1_id);
        REQUIRE(m2.id() == m2_id);
        REQUIRE(m3.id() == m3_id);
        REQUIRE(m4.id() == m4_id);
    }

    // --------------------------- Operator function ---------------------------
    SECTION("Function Arithmetic (NdArray, NdArray)") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(3.f);
        auto m_add = Add(m1, m2);
        auto m_sub = Subtract(m1, m2);
        auto m_mul = Multiply(m1, m2);
        auto m_div = Divide(m1, m2);
        RequireNdArray(m_add,
                       "[[0, 2, 4],\n"
                       " [3, 5, 7]]");
        RequireNdArray(m_sub,
                       "[[0, 0, 0],\n"
                       " [3, 3, 3]]");
        RequireNdArray(m_mul,
                       "[[0, 1, 4],\n"
                       " [0, 4, 10]]");
        // `0.f / 0.f` can be both of `nan` and `-nan`.
        m_div(0, 0) = std::abs(m_div(0, 0));
        RequireNdArray(m_div,
                       "[[nan, 1, 1],\n"
                       " [inf, 4, 2.5]]");
    }

    SECTION("Function Arithmetic (NdArray, float)") {
        auto m1 = NdArray::Arange(3.f);
        auto m_add = Add(m1, 2.f);
        auto m_sub = Subtract(m1, 2.f);
        auto m_mul = Multiply(m1, 2.f);
        auto m_div = Divide(m1, 2.f);
        RequireNdArray(m_add, "[2, 3, 4]");
        RequireNdArray(m_sub, "[-2, -1, 0]");
        RequireNdArray(m_mul, "[0, 2, 4]");
        RequireNdArray(m_div, "[0, 0.5, 1]");
    }

    SECTION("Function Arithmetic (float, NdArray)") {
        auto m1 = NdArray::Arange(3.f);
        auto m_add = Add(2.f, m1);
        auto m_sub = Subtract(2.f, m1);
        auto m_mul = Multiply(2.f, m1);
        auto m_div = Divide(2.f, m1);
        RequireNdArray(m_add, "[2, 3, 4]");
        RequireNdArray(m_sub, "[2, 1, 0]");
        RequireNdArray(m_mul, "[0, 2, 4]");
        RequireNdArray(m_div, "[inf, 2, 1]");
    }

    SECTION("Function Comparison") {
        auto m1 = NdArray::Arange(3.f);
        auto m2 = NdArray::Zeros(1) + 1.f;
        RequireNdArray(Equal(m1, m2), "[0, 1, 0]");
        RequireNdArray(NotEqual(m1, m2), "[1, 0, 1]");
        RequireNdArray(Greater(m1, m2), "[0, 0, 1]");
        RequireNdArray(GreaterEqual(m1, m2), "[0, 1, 1]");
        RequireNdArray(Less(m1, m2), "[1, 0, 0]");
        RequireNdArray(LessEqual(m1, m2), "[1, 1, 0]");
        RequireNdArray(Equal(m1, 1.f), "[0, 1, 0]");
        RequireNdArray(NotEqual(m1, 1.f), "[1, 0, 1]");
        RequireNdArray(Greater(m1, 1.f), "[0, 0, 1]");
        RequireNdArray(GreaterEqual(m1, 1.f), "[0, 1, 1]");
        RequireNdArray(Less(m1, 1.f), "[1, 0, 0]");
        RequireNdArray(LessEqual(m1, 1.f), "[1, 1, 0]");
        RequireNdArray(Equal(1.f, m1), "[0, 1, 0]");
        RequireNdArray(NotEqual(1.f, m1), "[1, 0, 1]");
        RequireNdArray(Greater(1.f, m1), "[1, 0, 0]");
        RequireNdArray(GreaterEqual(1.f, m1), "[1, 1, 0]");
        RequireNdArray(Less(1.f, m1), "[0, 0, 1]");
        RequireNdArray(LessEqual(1.f, m1), "[0, 1, 1]");
    }

    SECTION("Function Dot") {
        auto m1 = NdArray::Arange(6.f).reshape(2, 3);
        auto m2 = NdArray::Arange(3.f);
        auto m12 = Dot(m1, m2);
        auto m_a = Dot(m2, 2.f);
        auto m_b = Dot(2.f, m2);
        RequireNdArray(m12, "[5, 14]");
        RequireNdArray(m_a, "[0, 2, 4]");
        RequireNdArray(m_b, "[0, 2, 4]");
    }

    SECTION("Function Cross") {
        NdArray m1 = {1.f, 2.f, 3.f};
        NdArray m2 = {4.f, 5.f, 6.f};
        auto m12 = Cross(m1, m2);
        RequireNdArray(m12, "[-3, 6, -3]");
    }

    SECTION("Function Basic Math") {
        auto m1 = NdArray::Arange(3.f);
        auto m2 = NdArray::Arange(7.f) / 3.f - 1.f;
        RequireNdArray(-m1, "[-0, -1, -2]");
        RequireNdArray(Abs(-m1), "[0, 1, 2]");
        RequireNdArray(Ceil(m2), "[-1, -0, -0, 0, 1, 1, 1]");
        RequireNdArray(Floor(m2), "[-1, -1, -1, 0, 0, 0, 1]");
        RequireNdArray(Sqrt(m1), "[0, 1, 1.41421]");
        RequireNdArray(Exp(m1), "[1, 2.71828, 7.38906]");
        RequireNdArray(Log(m1), "[-inf, 0, 0.693147]");
        RequireNdArray(Power(m1, m1 + 1.f), "[0, 1, 8]");
        RequireNdArray(Power(m1, 4.f), "[0, 1, 16]");
        RequireNdArray(Power(4.f, m1), "[1, 4, 16]");
    }

    SECTION("Function Trigonometric") {
        auto m1 = NdArray::Arange(3.f);
        RequireNdArray(Sin(m1), "[0, 0.841471, 0.909297]");
        RequireNdArray(Cos(m1), "[1, 0.540302, -0.416147]");
        RequireNdArray(Tan(m1), "[0, 1.55741, -2.18504]");
    }

    SECTION("Function Inverse-Trigonometric") {
        auto m1 = NdArray::Arange(3) - 1.f;
        auto m2 = NdArray::Arange(3) * 100.f;
        RequireNdArray(ArcSin(m1), "[-1.5708, 0, 1.5708]");
        RequireNdArray(ArcCos(m1), "[3.14159, 1.5708, 0]");
        RequireNdArray(ArcTan(m2), "[0, 1.5608, 1.5658]");
        RequireNdArray(ArcTan2(m1, m2), "[-1.5708, 0, 0.00499996]");
        RequireNdArray(ArcTan2(m1, 2.f), "[-0.463648, 0, 0.463648]");
        RequireNdArray(ArcTan2(2.f, m1), "[2.03444, 1.5708, 1.10715]");
    }

    SECTION("Function Axis") {
        auto m1 = NdArray::Arange(36.f).reshape(2, 3, 2, 3);
        RequireNdArray(Sum(m1, {1, 3}),
                       "[[63, 90],\n"
                       " [225, 252]]");
        auto m2 = NdArray::Arange(12.f).reshape(2, 3, 2) - 6.f;
        RequireNdArray(Min(m2, {2, 1}), "[-6, 0]");
        RequireNdArray(Max(m2, {2, 1}), "[-1, 5]");
        RequireNdArray(Mean(m2, {2, 1}), "[-3.5, 2.5]");
    }

    SECTION("Function Inverse (2d)") {
        auto m1 = NdArray::Arange(4).reshape(2, 2) + 1.f;
        auto m2 = Inv(m1);
        RequireNdArray(m2,
                       "[[-2, 1],\n"
                       " [1.5, -0.5]]",
                       4);  // Low precision
        RequireNdArray(m1.dot(m2),
                       "[[1, 0],\n"
                       " [0, 1]]",
                       4);  // Low precision
    }

    SECTION("Function Inverse (high-dim)") {
        auto m1 = NdArray::Arange(24).reshape(2, 3, 2, 2) + 1.f;
        auto m2 = Inv(m1);
        RequireNdArray(m2,
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
        REQUIRE((m3.id() == m1_id || m3.id() == m2_id));  // m3 is not new array
        // Not in-place
        RequireNdArrayNotInplace(-NdArray::Arange(3.f), "[0, 1, 2]",
                                 static_cast<NdArray (*)(const NdArray&)>(Abs));
        // No matching broadcast
        NdArray m4 = NdArray::Arange(6).reshape(2, 1, 3);
        NdArray m5 = NdArray::Arange(2).reshape(2, 1);
        uintptr_t m4_id = m4.id();
        uintptr_t m5_id = m5.id();
        NdArray m6 = Power(std::move(m4), std::move(m5));
        REQUIRE((m6.id() != m4_id && m6.id() != m5_id));  // m6 is new array
    }

    SECTION("Function Arithmetic (NdArray, NdArray) (in-place both)") {
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f),
                "[[0, 2, 4],\n"
                " [3, 5, 7]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Add));
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f),
                "[[0, 0, 0],\n"
                " [3, 3, 3]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Subtract));
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f),
                "[[0, 1, 4],\n"
                " [0, 4, 10]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Multiply));
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), NdArray::Arange(3.f) + 1.f,
                "[[0, 0.5, 0.666667],\n"
                " [3, 2, 1.66667]]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Divide));
    }

    SECTION("Function Arithmetic (NdArray, NdArray) (in-place right)") {
        auto m1 = NdArray::Arange(3.f);
        auto m2 = NdArray::Arange(3.f) + 1.f;
        RequireNdArrayInplace(
                m1, NdArray::Arange(6.f).reshape(2, 3),
                "[[0, 2, 4],\n"
                " [3, 5, 7]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Add));
        RequireNdArrayInplace(
                m1, NdArray::Arange(6.f).reshape(2, 3),
                "[[0, 0, 0],\n"
                " [-3, -3, -3]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Subtract));
        RequireNdArrayInplace(
                m1, NdArray::Arange(6.f).reshape(2, 3),
                "[[0, 1, 4],\n"
                " [0, 4, 10]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Multiply));
        RequireNdArrayInplace(
                m2, NdArray::Arange(6.f).reshape(2, 3),
                "[[inf, 2, 1.5],\n"
                " [0.333333, 0.5, 0.6]]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Divide));
    }

    SECTION("Function Arithmetic (NdArray, NdArray) (in-place left)") {
        auto m2 = NdArray::Arange(3.f);
        auto m3 = NdArray::Arange(3.f) + 1.f;
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m2,
                "[[0, 2, 4],\n"
                " [3, 5, 7]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Add));
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m2,
                "[[0, 0, 0],\n"
                " [3, 3, 3]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Subtract));
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m2,
                "[[0, 1, 4],\n"
                " [0, 4, 10]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Multiply));
        RequireNdArrayInplace(
                NdArray::Arange(6.f).reshape(2, 3), m3,
                "[[0, 0.5, 0.666667],\n"
                " [3, 2, 1.66667]]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Divide));
    }

    SECTION("Function Arithmetic (NdArray, float) (inplace)") {
        RequireNdArrayInplace(NdArray::Arange(3.f), 2.f, "[2, 3, 4]",
                              static_cast<NdArray (*)(NdArray&&, float)>(Add));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 2.f, "[-2, -1, 0]",
                static_cast<NdArray (*)(NdArray&&, float)>(Subtract));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 2.f, "[0, 2, 4]",
                static_cast<NdArray (*)(NdArray&&, float)>(Multiply));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 2.f, "[0, 0.5, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(Divide));
    }

    SECTION("Function Arithmetic (float, NdArray) (inplace)") {
        RequireNdArrayInplace(2.f, NdArray::Arange(3.f), "[2, 3, 4]",
                              static_cast<NdArray (*)(float, NdArray&&)>(Add));
        RequireNdArrayInplace(
                2.f, NdArray::Arange(3.f), "[2, 1, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(Subtract));
        RequireNdArrayInplace(
                2.f, NdArray::Arange(3.f), "[0, 2, 4]",
                static_cast<NdArray (*)(float, NdArray&&)>(Multiply));
        RequireNdArrayInplace(
                2.f, NdArray::Arange(3.f), "[inf, 2, 1]",
                static_cast<NdArray (*)(float, NdArray&&)>(Divide));
    }

    SECTION("Function Comparison (NdArray, NdArray) (inplace both)") {
        RequireNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[0, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Equal));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[1, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(NotEqual));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[0, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Greater));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[0, 1, 1]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(GreaterEqual));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[1, 0, 0]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Less));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Zeros(1) + 1.f, "[1, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(LessEqual));
    }

    SECTION("Function Comparison (NdArray, NdArray) (inplace right)") {
        auto m2 = NdArray::Zeros(1) + 1.f;
        RequireNdArrayInplace(
                NdArray::Arange(3.f), m2, "[0, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Equal));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), m2, "[1, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(NotEqual));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), m2, "[0, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Greater));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), m2, "[0, 1, 1]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(
                        GreaterEqual));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), m2, "[1, 0, 0]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Less));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), m2, "[1, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(LessEqual));
    }

    SECTION("Function Comparison (NdArray, NdArray) (inplace left)") {
        auto m1 = NdArray::Zeros(1) + 1.f;
        RequireNdArrayInplace(
                m1, NdArray::Arange(3.f), "[0, 1, 0]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Equal));
        RequireNdArrayInplace(
                m1, NdArray::Arange(3.f), "[1, 0, 1]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(NotEqual));
        RequireNdArrayInplace(
                m1, NdArray::Arange(3.f), "[1, 0, 0]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Greater));
        RequireNdArrayInplace(
                m1, NdArray::Arange(3.f), "[1, 1, 0]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(
                        GreaterEqual));
        RequireNdArrayInplace(
                m1, NdArray::Arange(3.f), "[0, 0, 1]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Less));
        RequireNdArrayInplace(
                m1, NdArray::Arange(3.f), "[0, 1, 1]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(LessEqual));
    }

    SECTION("Function Comparison (NdArray, float) (inplace)") {
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[0, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, float)>(Equal));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[1, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(NotEqual));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[0, 0, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(Greater));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[0, 1, 1]",
                static_cast<NdArray (*)(NdArray&&, float)>(GreaterEqual));
        RequireNdArrayInplace(NdArray::Arange(3.f), 1.f, "[1, 0, 0]",
                              static_cast<NdArray (*)(NdArray&&, float)>(Less));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 1.f, "[1, 1, 0]",
                static_cast<NdArray (*)(NdArray&&, float)>(LessEqual));
    }

    SECTION("Function Comparison (NdArray, float) (inplace)") {
        RequireNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[0, 1, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(Equal));
        RequireNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[1, 0, 1]",
                static_cast<NdArray (*)(float, NdArray&&)>(NotEqual));
        RequireNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[1, 0, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(Greater));
        RequireNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[1, 1, 0]",
                static_cast<NdArray (*)(float, NdArray&&)>(GreaterEqual));
        RequireNdArrayInplace(1.f, NdArray::Arange(3.f), "[0, 0, 1]",
                              static_cast<NdArray (*)(float, NdArray&&)>(Less));
        RequireNdArrayInplace(
                1.f, NdArray::Arange(3.f), "[0, 1, 1]",
                static_cast<NdArray (*)(float, NdArray&&)>(LessEqual));
    }

    SECTION("Function Basic Math (in-place)") {
        RequireNdArrayInplace(-NdArray::Arange(3.f), "[0, 1, 2]",
                              static_cast<NdArray (*)(NdArray &&)>(Abs));
        RequireNdArrayInplace(NdArray::Arange(7.f) / 3.f - 1.f,
                              "[-1, -0, -0, 0, 1, 1, 1]",
                              static_cast<NdArray (*)(NdArray &&)>(Ceil));
        RequireNdArrayInplace(NdArray::Arange(7.f) / 3.f - 1.f,
                              "[-1, -1, -1, 0, 0, 0, 1]",
                              static_cast<NdArray (*)(NdArray &&)>(Floor));
        RequireNdArrayInplace(NdArray::Arange(3.f), "[0, 1, 1.41421]",
                              static_cast<NdArray (*)(NdArray &&)>(Sqrt));
        RequireNdArrayInplace(NdArray::Arange(3.f), "[1, 2.71828, 7.38906]",
                              static_cast<NdArray (*)(NdArray &&)>(Exp));
        RequireNdArrayInplace(NdArray::Arange(3.f), "[-inf, 0, 0.693147]",
                              static_cast<NdArray (*)(NdArray &&)>(Log));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), NdArray::Arange(3.f) + 1.f, "[0, 1, 8]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(Power));
        const auto m1 = NdArray::Arange(3.f) + 1.f;
        const auto m2 = NdArray::Arange(3.f);
        RequireNdArrayInplace(
                m2, NdArray::Arange(3.f) + 1.f, "[0, 1, 8]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(Power));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), m1, "[0, 1, 8]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(Power));
        RequireNdArrayInplace(
                NdArray::Arange(3.f), 4.f, "[0, 1, 16]",
                static_cast<NdArray (*)(NdArray&&, float)>(Power));
        RequireNdArrayInplace(
                4.f, NdArray::Arange(3.f), "[1, 4, 16]",
                static_cast<NdArray (*)(float, NdArray&&)>(Power));
    }

    SECTION("Function Trigonometric (in-place)") {
        RequireNdArrayInplace(NdArray::Arange(3.f), "[0, 0.841471, 0.909297]",
                              static_cast<NdArray (*)(NdArray &&)>(Sin));
        RequireNdArrayInplace(NdArray::Arange(3.f), "[1, 0.540302, -0.416147]",
                              static_cast<NdArray (*)(NdArray &&)>(Cos));
        RequireNdArrayInplace(NdArray::Arange(3.f), "[0, 1.55741, -2.18504]",
                              static_cast<NdArray (*)(NdArray &&)>(Tan));
    }

    SECTION("Function Inverse-Trigonometric (in-place)") {
        auto m1 = NdArray::Arange(3) - 1.f;
        auto m2 = NdArray::Arange(3) * 100.f;
        RequireNdArrayInplace(NdArray::Arange(3.f) - 1.f,
                              "[-1.5708, 0, 1.5708]",
                              static_cast<NdArray (*)(NdArray &&)>(ArcSin));
        RequireNdArrayInplace(NdArray::Arange(3.f) - 1.f,
                              "[3.14159, 1.5708, 0]",
                              static_cast<NdArray (*)(NdArray &&)>(ArcCos));
        RequireNdArrayInplace(NdArray::Arange(3.f) * 100.f,
                              "[0, 1.5608, 1.5658]",
                              static_cast<NdArray (*)(NdArray &&)>(ArcTan));
        RequireNdArrayInplace(
                NdArray::Arange(3.f) - 1.f, NdArray::Arange(3.f) * 100.f,
                "[-1.5708, 0, 0.00499996]",
                static_cast<NdArray (*)(NdArray&&, NdArray &&)>(ArcTan2));
        RequireNdArrayInplace(
                m1, NdArray::Arange(3.f) * 100.f, "[-1.5708, 0, 0.00499996]",
                static_cast<NdArray (*)(const NdArray&, NdArray&&)>(ArcTan2));
        RequireNdArrayInplace(
                NdArray::Arange(3.f) - 1.f, m2, "[-1.5708, 0, 0.00499996]",
                static_cast<NdArray (*)(NdArray&&, const NdArray&)>(ArcTan2));
        RequireNdArrayInplace(
                NdArray::Arange(3.f) - 1.f, 2.f, "[-0.463648, 0, 0.463648]",
                static_cast<NdArray (*)(NdArray&&, float)>(ArcTan2));
        RequireNdArrayInplace(
                2.f, NdArray::Arange(3.f) - 1.f, "[2.03444, 1.5708, 1.10715]",
                static_cast<NdArray (*)(float, NdArray&&)>(ArcTan2));
    }

    SECTION("Function Inverse (in-place)") {
        auto m1 = NdArray::Arange(4).reshape(2, 2) + 1.f;
        auto m2 = Inv(m1);
        RequireNdArrayInplace(NdArray::Arange(4).reshape(2, 2) + 1.f,
                              "[[-2, 1],\n"
                              " [1.5, -0.5]]",
                              static_cast<NdArray (*)(NdArray &&)>(Inv),
                              4);  // Low precision
    }
}
