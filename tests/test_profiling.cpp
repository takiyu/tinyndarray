#include "Catch2/single_include/catch2/catch.hpp"

#define TINYNDARRAY_IMPLEMENTATION
#define TINYNDARRAY_PROFILE_MEMORY  // With profiling
#include "../tinyndarray.h"

using tinyndarray::NdArray;

TEST_CASE("NdArray Profiling") {
    SECTION("Basic") {
        auto m1 = NdArray::Ones(10, 10);
        auto m2 = NdArray::Zeros(20, 20);
        CHECK(NdArray::GetNumInstance() == 2);
        CHECK(NdArray::GetTotalMemory() == (10 * 10) + (20 * 20));
    }

    SECTION("Unregister") {
        auto m1 = NdArray::Ones(10, 10);
        auto m2 = NdArray::Zeros(20, 20);
        {
            auto m3 = NdArray::Zeros(3, 3);
            CHECK(NdArray::GetNumInstance() == 3);
            CHECK(NdArray::GetTotalMemory() == (10 * 10) + (20 * 20) + (3 * 3));
        }

        CHECK(NdArray::GetNumInstance() == 2);
        CHECK(NdArray::GetTotalMemory() == (20 * 20) + (10 * 10));
    }

    SECTION("Overwrite") {
        auto m1 = NdArray::Ones(10, 10);
        auto m2 = NdArray::Zeros(20, 20);
        {
            auto m3 = NdArray::Zeros(3, 3);
            CHECK(NdArray::GetNumInstance() == 3);
            CHECK(NdArray::GetTotalMemory() == (10 * 10) + (20 * 20) + (3 * 3));

            m1 = m3;
            CHECK(NdArray::GetNumInstance() == 2);
            CHECK(NdArray::GetTotalMemory() == (20 * 20) + (3 * 3));
        }

        CHECK(NdArray::GetNumInstance() == 2);
        CHECK(NdArray::GetTotalMemory() == (20 * 20) + (3 * 3));
    }

    SECTION("Unregister of substance copy") {
        auto m1 = NdArray::Ones(10, 10);
        auto m2 = NdArray::Zeros(20, 20);
        {
            auto m3 = NdArray::Zeros(3, 3);
            CHECK(NdArray::GetNumInstance() == 3);
            CHECK(NdArray::GetTotalMemory() == (10 * 10) + (20 * 20) + (3 * 3));

            m3 = m1.reshape(-1);
            CHECK(NdArray::GetNumInstance() == 2);
            CHECK(NdArray::GetTotalMemory() == (10 * 10) + (20 * 20));
        }

        CHECK(NdArray::GetNumInstance() == 2);
        CHECK(NdArray::GetTotalMemory() == (10 * 10) + (20 * 20));
    }
}
