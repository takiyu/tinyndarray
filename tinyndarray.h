#ifndef TINYNDARRAY_H_ONCE
#define TINYNDARRAY_H_ONCE

#include <algorithm>
#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

namespace tinyndarray {

class NdArray;
using InitShape = std::initializer_list<int>;
using Shape = std::vector<int>;
using Index = std::vector<int>;
using SliceIndex = std::vector<std::pair<int, int>>;
using Axes = std::vector<int>;

// =============================================================================
// ======================= Nested Float Initializer List =======================
// =============================================================================
template <std::size_t D>
struct FloatListHelper {
    using type = std::initializer_list<typename FloatListHelper<D - 1>::type>;
};

template <>
struct FloatListHelper<0> {
    using type = std::initializer_list<float>;
};

template <std::size_t D>
using FloatList = typename FloatListHelper<D>::type;

// =============================================================================
// ================================== NdArray ==================================
// =============================================================================
class NdArray {
public:
    NdArray();
    NdArray(const NdArray&);
    NdArray(NdArray&&) noexcept;
    NdArray& operator=(const NdArray&);
    NdArray& operator=(NdArray&&);
    ~NdArray();

    NdArray(FloatList<0> init_list);
    NdArray(FloatList<1> init_list);
    NdArray(FloatList<2> init_list);
    NdArray(FloatList<3> init_list);
    NdArray(FloatList<4> init_list);
    NdArray(FloatList<5> init_list);
    NdArray(FloatList<6> init_list);
    NdArray(FloatList<7> init_list);
    NdArray(FloatList<8> init_list);
    NdArray(FloatList<9> init_list);

    NdArray(const InitShape& shape);
    NdArray(const Shape& shape);
    NdArray(const Shape& shape, float fill_v);

    static NdArray Empty(const Shape& shape);
    static NdArray Zeros(const Shape& shape);
    static NdArray Ones(const Shape& shape);
    template <typename... S>
    static NdArray Empty(S... shape);
    template <typename... S>
    static NdArray Zeros(S... shape);
    template <typename... S>
    static NdArray Ones(S... shape);

    static NdArray Arange(float stop);
    static NdArray Arange(float start, float stop, float step = 1.f);

    static void Seed();
    static void Seed(uint32_t seed);
    static NdArray Uniform(float low = 0.f, float high = 1.f,
                           const Shape& shape = {1});
    static NdArray Uniform(const Shape& shape);
    static NdArray Normal(float loc = 0.f, float scale = 1.f,
                          const Shape& shape = {1});
    static NdArray Normal(const Shape& shape);

    uintptr_t id() const;
    bool empty() const;
    size_t size() const;
    const Shape& shape() const;
    size_t ndim() const;
    float* data();
    const float* data() const;
    void fill(float v);
    NdArray copy() const;

    float* begin();
    float* end();
    const float* begin() const;
    const float* end() const;

    operator float() const;

    float& operator[](int i);
    const float& operator[](int i) const;

    float& operator[](const Index& index);
    const float& operator[](const Index& index) const;
    template <typename... I>
    float& operator()(I... index);
    template <typename... I>
    const float& operator()(I... index) const;

    NdArray reshape(const Shape& shape) const;
    template <typename... S>
    NdArray reshape(S... shape) const;
    NdArray flatten() const;  // with copy
    NdArray ravel() const;    // without copy

    NdArray slice(const SliceIndex& slice_index) const;
    template <typename... I>
    NdArray slice(std::initializer_list<I>... slice_index) const;  // {i, j}...

    NdArray dot(const NdArray& other) const;
    NdArray dot(float other) const;
    NdArray cross(const NdArray& other) const;

    NdArray sum(const Axes& = {}) const;
    NdArray min(const Axes& = {}) const;
    NdArray max(const Axes& = {}) const;
    NdArray mean(const Axes& = {}) const;

    class Substance;

private:
    std::shared_ptr<Substance> m_sub;
    NdArray(std::shared_ptr<Substance> sub);

    static std::random_device s_rand_seed;
    static std::mt19937 s_rand_engine;
};

// --------------------------------- Operators ---------------------------------
// Print
std::ostream& operator<<(std::ostream& os, const NdArray& x);
std::ostream& operator<<(std::ostream& os, const Shape& shape);
// Single
NdArray operator+(const NdArray& x);
NdArray operator-(const NdArray& x);
// Arithmetic (NdArray, NdArray)
NdArray operator+(const NdArray& lhs, const NdArray& rhs);
NdArray operator-(const NdArray& lhs, const NdArray& rhs);
NdArray operator*(const NdArray& lhs, const NdArray& rhs);
NdArray operator/(const NdArray& lhs, const NdArray& rhs);
// Arithmetic (NdArray, float)
NdArray operator+(const NdArray& lhs, const float& rhs);
NdArray operator-(const NdArray& lhs, const float& rhs);
NdArray operator*(const NdArray& lhs, const float& rhs);
NdArray operator/(const NdArray& lhs, const float& rhs);
// Arithmetic (float, NdArray)
NdArray operator+(const float& lhs, const NdArray& rhs);
NdArray operator-(const float& lhs, const NdArray& rhs);
NdArray operator*(const float& lhs, const NdArray& rhs);
NdArray operator/(const float& lhs, const NdArray& rhs);
// Comparison (NdArray, NdArray)
NdArray operator==(const NdArray& lhs, const NdArray& rhs);
NdArray operator!=(const NdArray& lhs, const NdArray& rhs);
NdArray operator>(const NdArray& lhs, const NdArray& rhs);
NdArray operator>=(const NdArray& lhs, const NdArray& rhs);
NdArray operator<(const NdArray& lhs, const NdArray& rhs);
NdArray operator<=(const NdArray& lhs, const NdArray& rhs);
// Comparison (NdArray, float)
NdArray operator==(const NdArray& lhs, float rhs);
NdArray operator!=(const NdArray& lhs, float rhs);
NdArray operator>(const NdArray& lhs, float rhs);
NdArray operator>=(const NdArray& lhs, float rhs);
NdArray operator<(const NdArray& lhs, float rhs);
NdArray operator<=(const NdArray& lhs, float rhs);
// Comparison (float, NdArray)
NdArray operator==(float lhs, const NdArray& rhs);
NdArray operator!=(float lhs, const NdArray& rhs);
NdArray operator>(float lhs, const NdArray& rhs);
NdArray operator>=(float lhs, const NdArray& rhs);
NdArray operator<(float lhs, const NdArray& rhs);
NdArray operator<=(float lhs, const NdArray& rhs);
// ----------------------------- In-place Operators ----------------------------
// Single
NdArray operator+(NdArray&& x);
NdArray operator-(NdArray&& x);
// Arithmetic (NdArray, NdArray)
NdArray operator+(NdArray&& lhs, NdArray&& rhs);
NdArray operator+(const NdArray& lhs, NdArray&& rhs);
NdArray operator+(NdArray&& lhs, const NdArray& rhs);
NdArray operator-(NdArray&& lhs, NdArray&& rhs);
NdArray operator-(const NdArray& lhs, NdArray&& rhs);
NdArray operator-(NdArray&& lhs, const NdArray& rhs);
NdArray operator*(NdArray&& lhs, NdArray&& rhs);
NdArray operator*(const NdArray& lhs, NdArray&& rhs);
NdArray operator*(NdArray&& lhs, const NdArray& rhs);
NdArray operator/(NdArray&& lhs, NdArray&& rhs);
NdArray operator/(const NdArray& lhs, NdArray&& rhs);
NdArray operator/(NdArray&& lhs, const NdArray& rhs);
// Arithmetic (NdArray, float)
NdArray operator+(NdArray&& lhs, float rhs);
NdArray operator-(NdArray&& lhs, float rhs);
NdArray operator*(NdArray&& lhs, float rhs);
NdArray operator/(NdArray&& lhs, float rhs);
// Arithmetic (float, NdArray)
NdArray operator+(float lhs, NdArray&& rhs);
NdArray operator-(float lhs, NdArray&& rhs);
NdArray operator*(float lhs, NdArray&& rhs);
NdArray operator/(float lhs, NdArray&& rhs);
// Comparison (NdArray, NdArray)
NdArray operator==(NdArray&& lhs, NdArray&& rhs);
NdArray operator==(const NdArray& lhs, NdArray&& rhs);
NdArray operator==(NdArray&& lhs, const NdArray& rhs);
NdArray operator!=(NdArray&& lhs, NdArray&& rhs);
NdArray operator!=(const NdArray& lhs, NdArray&& rhs);
NdArray operator!=(NdArray&& lhs, const NdArray& rhs);
NdArray operator>(NdArray&& lhs, NdArray&& rhs);
NdArray operator>(const NdArray& lhs, NdArray&& rhs);
NdArray operator>(NdArray&& lhs, const NdArray& rhs);
NdArray operator>=(NdArray&& lhs, NdArray&& rhs);
NdArray operator>=(const NdArray& lhs, NdArray&& rhs);
NdArray operator>=(NdArray&& lhs, const NdArray& rhs);
NdArray operator<(NdArray&& lhs, NdArray&& rhs);
NdArray operator<(const NdArray& lhs, NdArray&& rhs);
NdArray operator<(NdArray&& lhs, const NdArray& rhs);
NdArray operator<=(NdArray&& lhs, NdArray&& rhs);
NdArray operator<=(const NdArray& lhs, NdArray&& rhs);
NdArray operator<=(NdArray&& lhs, const NdArray& rhs);
// Comparison (NdArray, float)
NdArray operator==(NdArray&& lhs, float rhs);
NdArray operator!=(NdArray&& lhs, float rhs);
NdArray operator>(NdArray&& lhs, float rhs);
NdArray operator>=(NdArray&& lhs, float rhs);
NdArray operator<(NdArray&& lhs, float rhs);
NdArray operator<=(NdArray&& lhs, float rhs);
// Comparison (float, NdArray)
NdArray operator==(float lhs, NdArray&& rhs);
NdArray operator!=(float lhs, NdArray&& rhs);
NdArray operator>(float lhs, NdArray&& rhs);
NdArray operator>=(float lhs, NdArray&& rhs);
NdArray operator<(float lhs, NdArray&& rhs);
NdArray operator<=(float lhs, NdArray&& rhs);
// Compound Assignment (NdArray, NdArray)
NdArray operator+=(NdArray& lhs, const NdArray& rhs);
NdArray operator-=(NdArray& lhs, const NdArray& rhs);
NdArray operator*=(NdArray& lhs, const NdArray& rhs);
NdArray operator/=(NdArray& lhs, const NdArray& rhs);
// Compound Assignment (NdArray, float)
NdArray operator+=(NdArray& lhs, float rhs);
NdArray operator-=(NdArray& lhs, float rhs);
NdArray operator*=(NdArray& lhs, float rhs);
NdArray operator/=(NdArray& lhs, float rhs);

// ---------------------------- Operator Functions -----------------------------
// Arithmetic operators (NdArray, NdArray)
NdArray Add(const NdArray& lhs, const NdArray& rhs);
NdArray Subtract(const NdArray& lhs, const NdArray& rhs);
NdArray Multiply(const NdArray& lhs, const NdArray& rhs);
NdArray Divide(const NdArray& lhs, const NdArray& rhs);
// Arithmetic operators (NdArray, float)
NdArray Add(const NdArray& lhs, float rhs);
NdArray Subtract(const NdArray& lhs, float rhs);
NdArray Multiply(const NdArray& lhs, float rhs);
NdArray Divide(const NdArray& lhs, float rhs);
// Arithmetic operators (float, NdArray)
NdArray Add(float lhs, const NdArray& rhs);
NdArray Subtract(float lhs, const NdArray& rhs);
NdArray Multiply(float lhs, const NdArray& rhs);
NdArray Divide(float lhs, const NdArray& rhs);
// Comparison operators (NdArray, NdArray)
NdArray Equal(const NdArray& lhs, const NdArray& rhs);
NdArray NotEqual(const NdArray& lhs, const NdArray& rhs);
NdArray Greater(const NdArray& lhs, const NdArray& rhs);       // >
NdArray GreaterEqual(const NdArray& lhs, const NdArray& rhs);  // >=
NdArray Less(const NdArray& lhs, const NdArray& rhs);          // <
NdArray LessEqual(const NdArray& lhs, const NdArray& rhs);     // <=
// Comparison operators (NdArray, float)
NdArray Equal(const NdArray& lhs, float rhs);
NdArray NotEqual(const NdArray& lhs, float rhs);
NdArray Greater(const NdArray& lhs, float rhs);
NdArray GreaterEqual(const NdArray& lhs, float rhs);
NdArray Less(const NdArray& lhs, float rhs);
NdArray LessEqual(const NdArray& lhs, float rhs);
// Comparison operators (float, NdArray)
NdArray Equal(float lhs, const NdArray& rhs);
NdArray NotEqual(float lhs, const NdArray& rhs);
NdArray Greater(float lhs, const NdArray& rhs);
NdArray GreaterEqual(float lhs, const NdArray& rhs);
NdArray Less(float lhs, const NdArray& rhs);
NdArray LessEqual(float lhs, const NdArray& rhs);
// Matrix operators
NdArray Dot(const NdArray& lhs, const NdArray& rhs);
NdArray Dot(const NdArray& lhs, float rhs);
NdArray Dot(float lhs, const NdArray& rhs);
NdArray Cross(const NdArray& lhs, const NdArray& rhs);
// Basic math operators
NdArray Abs(const NdArray& x);
NdArray Ceil(const NdArray& x);
NdArray Floor(const NdArray& x);
NdArray Sqrt(const NdArray& x);
NdArray Exp(const NdArray& x);
NdArray Log(const NdArray& x);
NdArray Power(const NdArray& x, const NdArray& y);
NdArray Power(const NdArray& x, float y);
NdArray Power(float x, const NdArray& y);
// Trigonometric functions
NdArray Sin(const NdArray& x);
NdArray Cos(const NdArray& x);
NdArray Tan(const NdArray& x);
// Inverse trigonometric functions
NdArray ArcSin(const NdArray& x);
NdArray ArcCos(const NdArray& x);
NdArray ArcTan(const NdArray& x);
NdArray ArcTan2(const NdArray& y, const NdArray& x);
NdArray ArcTan2(const NdArray& y, float x);
NdArray ArcTan2(float y, const NdArray& x);
// Axis functions
NdArray Sum(const NdArray& x, const Axes& axes = {});
NdArray Min(const NdArray& x, const Axes& axes = {});
NdArray Max(const NdArray& x, const Axes& axes = {});
NdArray Mean(const NdArray& x, const Axes& axes = {});
// Inverse
NdArray Inv(const NdArray& x);
// ------------------------ In-place Operator Functions ------------------------
// Arithmetic operators (NdArray, NdArray)
NdArray Add(NdArray&& lhs, NdArray&& rhs);
NdArray Add(const NdArray& lhs, NdArray&& rhs);
NdArray Add(NdArray&& lhs, const NdArray& rhs);
NdArray Subtract(NdArray&& lhs, NdArray&& rhs);
NdArray Subtract(const NdArray& lhs, NdArray&& rhs);
NdArray Subtract(NdArray&& lhs, const NdArray& rhs);
NdArray Multiply(NdArray&& lhs, NdArray&& rhs);
NdArray Multiply(const NdArray& lhs, NdArray&& rhs);
NdArray Multiply(NdArray&& lhs, const NdArray& rhs);
NdArray Divide(NdArray&& lhs, NdArray&& rhs);
NdArray Divide(const NdArray& lhs, NdArray&& rhs);
NdArray Divide(NdArray&& lhs, const NdArray& rhs);
// Arithmetic operators (NdArray, float)
NdArray Add(NdArray&& lhs, float rhs);
NdArray Subtract(NdArray&& lhs, float rhs);
NdArray Multiply(NdArray&& lhs, float rhs);
NdArray Divide(NdArray&& lhs, float rhs);
// Arithmetic operators (float, NdArray)
NdArray Add(float lhs, NdArray&& rhs);
NdArray Subtract(float lhs, NdArray&& rhs);
NdArray Multiply(float lhs, NdArray&& rhs);
NdArray Divide(float lhs, NdArray&& rhs);
// Comparison operators (NdArray, NdArray)
NdArray Equal(NdArray&& lhs, NdArray&& rhs);
NdArray Equal(const NdArray& lhs, NdArray&& rhs);
NdArray Equal(NdArray&& lhs, const NdArray& rhs);
NdArray NotEqual(NdArray&& lhs, NdArray&& rhs);
NdArray NotEqual(const NdArray& lhs, NdArray&& rhs);
NdArray NotEqual(NdArray&& lhs, const NdArray& rhs);
NdArray Greater(NdArray&& lhs, NdArray&& rhs);
NdArray Greater(const NdArray& lhs, NdArray&& rhs);
NdArray Greater(NdArray&& lhs, const NdArray& rhs);
NdArray GreaterEqual(NdArray&& lhs, NdArray&& rhs);
NdArray GreaterEqual(const NdArray& lhs, NdArray&& rhs);
NdArray GreaterEqual(NdArray&& lhs, const NdArray& rhs);
NdArray Less(NdArray&& lhs, NdArray&& rhs);
NdArray Less(const NdArray& lhs, NdArray&& rhs);
NdArray Less(NdArray&& lhs, const NdArray& rhs);
NdArray LessEqual(NdArray&& lhs, NdArray&& rhs);
NdArray LessEqual(const NdArray& lhs, NdArray&& rhs);
NdArray LessEqual(NdArray&& lhs, const NdArray& rhs);
// Comparison operators (NdArray, float)
NdArray Equal(NdArray&& lhs, float rhs);
NdArray NotEqual(NdArray&& lhs, float rhs);
NdArray Greater(NdArray&& lhs, float rhs);
NdArray GreaterEqual(NdArray&& lhs, float rhs);
NdArray Less(NdArray&& lhs, float rhs);
NdArray LessEqual(NdArray&& lhs, float rhs);
// Comparison operators (float, NdArray)
NdArray Equal(float lhs, NdArray&& rhs);
NdArray NotEqual(float lhs, NdArray&& rhs);
NdArray Greater(float lhs, NdArray&& rhs);
NdArray GreaterEqual(float lhs, NdArray&& rhs);
NdArray Less(float lhs, NdArray&& rhs);
NdArray LessEqual(float lhs, NdArray&& rhs);
// Basic math operators
NdArray Abs(NdArray&& x);
NdArray Ceil(NdArray&& x);
NdArray Floor(NdArray&& x);
NdArray Sqrt(NdArray&& x);
NdArray Exp(NdArray&& x);
NdArray Log(NdArray&& x);
NdArray Power(NdArray&& x, NdArray&& y);
NdArray Power(const NdArray& x, NdArray&& y);
NdArray Power(NdArray&& x, const NdArray& y);
NdArray Power(NdArray&& x, float y);
NdArray Power(float x, NdArray&& y);
// Trigonometric functions
NdArray Sin(NdArray&& x);
NdArray Cos(NdArray&& x);
NdArray Tan(NdArray&& x);
// Inverse trigonometric functions
NdArray ArcSin(NdArray&& x);
NdArray ArcCos(NdArray&& x);
NdArray ArcTan(NdArray&& x);
NdArray ArcTan2(NdArray&& y, NdArray&& x);
NdArray ArcTan2(const NdArray& y, NdArray&& x);
NdArray ArcTan2(NdArray&& y, const NdArray& x);
NdArray ArcTan2(NdArray&& y, float x);
NdArray ArcTan2(float y, NdArray&& x);
// Inverse
NdArray Inv(NdArray&& x);

// *****************************************************************************
// *****************************************************************************
// **************************** Begin of Definitions ***************************
// *****************************************************************************
// *****************************************************************************
#ifdef TINYNDARRAY_IMPLEMENTATION

// -----------------------------------------------------------------------------
// --------------------------- Utilities for NdArray ---------------------------
// -----------------------------------------------------------------------------
template <typename T>
T Clamp(const T& v, const T& lower, const T& upper) {
    return std::min(std::max(v, lower), upper);
}

template <typename F>
inline auto ReverseOp(F op) {
    return [op](float a, float b) { return op(b, a); };  // Swap left and right
}

template <typename F>
inline void ApplyOpSimple(NdArray& ret, F op) {
    float* ret_data = ret.data();
    // Simply apply all
    for (size_t i = 0; i < ret.size(); i++) {
        *(ret_data++) = op();
    }
}

template <typename F>
inline void ApplyOpSimple(NdArray& ret, const NdArray& src, F op) {
    float* ret_data = ret.data();
    const float* src_data = src.data();
    // Simply apply all
    for (size_t i = 0; i < ret.size(); i++) {
        *(ret_data++) = op(*(src_data++));
    }
}

template <typename F>
inline void ApplyOpSimple(NdArray& ret, const NdArray& lhs, const NdArray& rhs,
                          F op) {
    float* ret_data = ret.data();
    const float* l_data = lhs.data();
    const float* r_data = rhs.data();
    // Simply apply all
    for (size_t i = 0; i < ret.size(); i++) {
        *(ret_data++) = op(*(l_data++), *(r_data++));
    }
}

template <typename F>
inline void ApplyOpSimple(NdArray& ret, const NdArray& lhs, const float rhs,
                          F op) {
    float* ret_data = ret.data();
    const float* l_data = lhs.data();
    // Simply apply all
    for (size_t i = 0; i < ret.size(); i++) {
        *(ret_data++) = op(*(l_data++), rhs);
    }
}

static std::vector<int> ComputeChildSizes(const Shape& shape) {
    const size_t n_shape = shape.size();
    if (n_shape == 0) {
        return {};
    }
    // Compute child sizes from back (the number of children for each dimension)
    std::vector<int> child_sizes(n_shape, 1);
    int size = 1;
    for (size_t depth = n_shape - 1; 0 < depth; depth--) {
        child_sizes[depth] = size;
        size *= shape[depth];
    }
    child_sizes[0] = size;
    return child_sizes;
}

// --------------- Utilities for NdArray (Float initializer list) --------------
template <typename FList>
std::list<int> CheckFListShapeImpl(const FList& init_list) {
    if (init_list.size() == 0) {
        return {};
    }
    // Check all children have same shape
    auto itr = init_list.begin();
    auto shape = CheckFListShapeImpl(*itr);
    for (size_t i = 0; i < init_list.size(); i++, itr++) {
        if (shape != CheckFListShapeImpl(*itr)) {
            throw std::runtime_error("Initializing shape is invalid");
        }
    }
    // Return total shape of children
    shape.push_front(static_cast<int>(init_list.size()));
    return shape;
}

template <>
inline std::list<int> CheckFListShapeImpl(const FloatList<0>& init_list) {
    return {static_cast<int>(init_list.size())};
}

template <typename FList>
Shape CheckFListShape(const FList& init_list) {
    // Check and get the shape of nested initializer.
    const std::list<int>& shape = CheckFListShapeImpl(init_list);
    // Cast to vector
    return Shape(shape.begin(), shape.end());
}

template <typename FList>
void CopyFListElemsImpl(const FList& init_list, float*& data) {
    // Copy sequentially
    for (auto itr = init_list.begin(); itr != init_list.end(); itr++) {
        CopyFListElemsImpl(*itr, data);
    }
}

template <>
void CopyFListElemsImpl(const FloatList<0>& init_list, float*& data) {
    // Copy sequentially
    for (auto&& v : init_list) {
        *(data++) = v;
    }
}

template <typename FList>
void CopyFListElems(const FList& init_list, float* data) {
    // Pass to impl (create pointer instance)
    CopyFListElemsImpl(init_list, data);
}

// ---------------------- Utilities for NdArray (Random) -----------------------
template <typename D, typename R>
NdArray CreateRandomArray(const Shape& shape, D&& dist, R&& rand_engine) {
    // Create empty array
    NdArray ret(shape);
    // Fill by random value
    ApplyOpSimple(ret, [&]() { return static_cast<float>(dist(rand_engine)); });
    return ret;
}

// ----------------------- Utilities for NdArray (Slice) -----------------------
static void CopySliceImpl(const float*& src_data, float*& dst_data,
                          const Shape& src_shape, const SliceIndex& slice_index,
                          const std::vector<int>& child_sizes, size_t depth) {
    if (depth < src_shape.size()) {
        const auto& si = slice_index[depth];
        // Add previous offset
        src_data += child_sizes[depth] * si.first;
        // Copy
        for (int i = si.first; i < si.second; i++) {
            // Recursive call
            CopySliceImpl(src_data, dst_data, src_shape, slice_index,
                          child_sizes, depth + 1);
        }
        // Add post offset
        src_data += child_sizes[depth] * (src_shape[depth] - si.second);
    } else {
        // Copy
        *(dst_data++) = *(src_data++);
    }
}

static NdArray CopySlice(const NdArray& src, const Shape& slice_shape,
                         const SliceIndex& slice_index) {
    const Shape& src_shape = src.shape();

    // Pre-compute child sizes (index offsets)
    const std::vector<int>& child_sizes = ComputeChildSizes(src_shape);

    // Create slice instance
    NdArray ret(slice_shape);

    // Start to copy
    const float* src_data = src.data();
    float* dst_data = ret.data();
    CopySliceImpl(src_data, dst_data, src_shape, slice_index, child_sizes, 0);

    return ret;
}

inline std::pair<int, int> CvtToSliceIndexItem(std::initializer_list<int> l) {
    if (l.size() != 2) {
        throw std::runtime_error("Invalid slice index format");
    }
    return {*l.begin(), *(l.begin() + 1)};
}

// ------------------ Utilities for NdArray (Single operator) ------------------
template <typename F>
NdArray ApplySingleOp(const NdArray& x, F op) {
    NdArray ret(x.shape());
    ApplyOpSimple(ret, x, op);
    return ret;
}

template <typename F>
NdArray ApplySingleOpInplace(NdArray&& x, F op) {
    ApplyOpSimple(x, x, op);
    return x;
}

// ------------------ Utilities for NdArray (Broadcast common) -----------------
static Shape CheckBroadcastable(const Shape& l_shape, const Shape& r_shape) {
    // We assuming left array has deeper shape than right one.
    if (l_shape.size() < r_shape.size()) {
        return CheckBroadcastable(r_shape, l_shape);  // Swap
    }
    // `l_shape.size()` is maximum depth.

    // Compute broadcasted shape
    Shape shape(l_shape.size());
    size_t r_offset = l_shape.size() - r_shape.size();
    for (size_t i = 0; i < l_shape.size(); i++) {
        if (i < r_offset) {
            shape[i] = l_shape[i];
        } else {
            const int l = l_shape[i];
            const int r = r_shape[i - r_offset];
            if (l == r) {
                shape[i] = l;  // no broadcast
            } else if (l == 1) {
                shape[i] = r;  // left broadcast
            } else if (r == 1) {
                shape[i] = l;  // right broadcast
            } else {
                std::stringstream ss;
                ss << "Non operatable shape";
                ss << " (" << l_shape << " vs " << r_shape << ")";
                throw std::runtime_error(ss.str());
            }
        }
    }
    return shape;
}

static Shape PadShape(const Shape& shape, size_t size) {
    if (size < shape.size()) {
        throw std::runtime_error("Invalid shape to pad");
    }
    const size_t n_pad = size - shape.size();
    Shape ret_shape;
    ret_shape.reserve(size);
    ret_shape.resize(n_pad, 1);                                     // Fill by 1
    ret_shape.insert(ret_shape.end(), shape.begin(), shape.end());  // Concat
    return ret_shape;
}

template <typename F>
void ApplyOpBroadcastImpl(float* ret_data, const float* l_data,
                          const float* r_data, const Shape& ret_shape,
                          const Shape& l_shape, const Shape& r_shape,
                          const std::vector<int>& ret_child_sizes,
                          const std::vector<int>& l_child_sizes,
                          const std::vector<int>& r_child_sizes, size_t depth,
                          size_t const depth_offset, F op) {
    if (depth < ret_shape.size() - depth_offset) {
        // Fetch shapes
        const int l_s = l_shape[depth];
        const int r_s = r_shape[depth];
        // Decide pointer steps by broadcast patterns.
        const int ret_step = ret_child_sizes[depth];
        const int l_step = (l_s == r_s || r_s == 1) ? l_child_sizes[depth] : 0;
        const int r_step = (l_s == r_s || l_s == 1) ? r_child_sizes[depth] : 0;
        // Applying loop
        const int n_loop = std::max(l_s, r_s);
        for (int i = 0; i < n_loop; i++) {
            // Apply recursively
            ApplyOpBroadcastImpl(ret_data, l_data, r_data, ret_shape, l_shape,
                                 r_shape, ret_child_sizes, l_child_sizes,
                                 r_child_sizes, depth + 1, depth_offset, op);
            // Next pointer
            ret_data += ret_step;
            l_data += l_step;
            r_data += r_step;
        }
    } else {
        // Apply operator
        op(ret_data, l_data, r_data);
    }
}

template <typename F>
void ApplyOpBroadcast(NdArray& ret, const NdArray& lhs, const NdArray& rhs,
                      const size_t depth_offset, F op) {
    const Shape& ret_shape = ret.shape();

    // Pre-compute padded shape
    const Shape& l_shape_pad = PadShape(lhs.shape(), ret_shape.size());
    const Shape& r_shape_pad = PadShape(rhs.shape(), ret_shape.size());

    // Pre-compute child sizes
    const std::vector<int>& ret_child_sizes = ComputeChildSizes(ret_shape);
    const std::vector<int>& l_child_sizes = ComputeChildSizes(l_shape_pad);
    const std::vector<int>& r_child_sizes = ComputeChildSizes(r_shape_pad);

    // Apply with broadcast
    ApplyOpBroadcastImpl(ret.data(), lhs.data(), rhs.data(), ret_shape,
                         l_shape_pad, r_shape_pad, ret_child_sizes,
                         l_child_sizes, r_child_sizes, 0, depth_offset, op);
}

// --------------- Utilities for NdArray (Broadcast element-wise) --------------
template <typename F>
NdArray ApplyElemWiseOp(const NdArray& lhs, const NdArray& rhs, F op) {
    if (lhs.shape() == rhs.shape()) {
        // Apply without broadcast because of same size for speed up.
        NdArray ret(lhs.shape());
        // Simply apply all
        ApplyOpSimple(ret, lhs, rhs, op);
        return ret;
    } else {
        // Check it is possible to broadcast
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        // Apply broadcast
        NdArray ret(ret_shape);
        ApplyOpBroadcast(ret, lhs, rhs, 0,
                         [&](float* o, const float* l, const float* r) {
                             *o = op(*l, *r);  // wrap for pointer operation
                         });
        return ret;
    }
}

template <typename F>
NdArray ApplyElemWiseOp(const NdArray& lhs, const float rhs, F op) {
    // Broadcast right float
    NdArray ret(lhs.shape());
    // Simply apply all
    ApplyOpSimple(ret, lhs, rhs, op);
    return ret;
}

template <typename F>
inline NdArray ApplyElemWiseOp(const float lhs, const NdArray& rhs, F op) {
    // Swap left and right
    return ApplyElemWiseOp(rhs, lhs, ReverseOp(op));
}

// ---------- Utilities for NdArray (Broadcast element-wise in-place) ----------
template <typename F>
NdArray ApplyElemWiseOpInplace(NdArray&& lhs, NdArray&& rhs, F op,
                               const bool allow_new = true) {
    if (lhs.shape() == rhs.shape()) {
        // Apply without broadcast because of same size for speed up.
        ApplyOpSimple(lhs, lhs, rhs, op);  // Use left as result
        return lhs;
    } else {
        // Check it is possible to broadcast
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        // Select result with shape matching
        NdArray&& ret = (ret_shape == lhs.shape()) ? lhs :          // left
                                (ret_shape == rhs.shape()) ? rhs :  // right
                                        (allow_new) ? NdArray(ret_shape)
                                                    :  // new instance
                                                throw std::runtime_error(
                                                        "Invalid shape for "
                                                        "in-place operation");
        // Apply broadcast
        ApplyOpBroadcast(ret, lhs, rhs, 0,
                         [&](float* o, const float* l, const float* r) {
                             *o = op(*l, *r);  // wrap for pointer operation
                         });
        return ret;
    }
}

template <typename F>
NdArray ApplyElemWiseOpInplace(NdArray&& lhs, const NdArray& rhs, F op,
                               const bool allow_new = true) {
    if (lhs.shape() == rhs.shape()) {
        // Apply without broadcast because of same size for speed up.
        ApplyOpSimple(lhs, lhs, rhs, op);
        return lhs;
    } else {
        // Check it is possible to broadcast
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        // Select result with shape matching
        NdArray&& ret =
                (ret_shape == lhs.shape()) ? lhs :          // left
                        (allow_new) ? NdArray(ret_shape) :  // new instance
                                throw std::runtime_error(
                                        "Invalid shape for in-place operation");
        // Apply broadcast (result matrix is lhs)
        ApplyOpBroadcast(ret, lhs, rhs, 0,
                         [&](float* o, const float* l, const float* r) {
                             *o = op(*l, *r);  // wrap for pointer operation
                         });
        return ret;
    }
}

template <typename F>
inline NdArray ApplyElemWiseOpInplace(const NdArray& lhs, NdArray&& rhs, F op,
                                      const bool allow_new = true) {
    // Swap left and right
    return ApplyElemWiseOpInplace(std::move(rhs), lhs, ReverseOp(op),
                                  allow_new);
}

template <typename F>
NdArray ApplyElemWiseOpInplace(NdArray&& lhs, float rhs, F op) {
    // Broadcast right float
    // Simply apply all
    ApplyOpSimple(lhs, lhs, rhs, op);
    return lhs;
}

template <typename F>
inline NdArray ApplyElemWiseOpInplace(float lhs, NdArray&& rhs, F op) {
    // Swap left and right
    return ApplyElemWiseOpInplace(std::move(rhs), lhs, ReverseOp(op));
}

// ------------------- Utilities for NdArray (Axis reduction) ------------------
static int ComputeReducedIndex(int src_idx,
                               const std::vector<int>& ret_child_sizes,
                               const std::vector<int>& src_child_sizes,
                               const Axes& sorted_axes) {
    // Convert source index to result index
    // [2, (3), 4, (5), 6]
    int ret_idx = 0;
    for (auto&& axis : sorted_axes) {
        if (axis == 0) {
            continue;  // No upper dimension
        }
        const size_t axis_l = static_cast<size_t>(axis);
        // Accumulate upper dimension
        const int ret_idx_base = src_idx / src_child_sizes[axis_l - 1];
        ret_idx += ret_idx_base * ret_child_sizes[axis_l];
        // Remove processed dimension
        src_idx = src_idx % src_child_sizes[axis_l];
    }

    // Add rest dimension
    const int last_axis = sorted_axes.back();
    ret_idx += src_idx % src_child_sizes[static_cast<size_t>(last_axis)];

    return ret_idx;
}

static auto CheckReductable(const Shape& shape, const Axes& axes) {
    // Mark reduction axes
    std::vector<char> mark(shape.size(), false);
    const int n_shape = static_cast<int>(shape.size());
    for (auto&& axis : axes) {
        if (0 <= axis && axis < n_shape) {
            mark[static_cast<size_t>(axis)] = true;
        } else {
            throw std::runtime_error("Invalid axes for reduction");
        }
    }

    // Pick up unmarked dimension
    Shape ret_shape;
    Shape ret_shape_pad;
    for (size_t i = 0; i < mark.size(); i++) {
        if (mark[i]) {
            ret_shape_pad.push_back(1);
        } else {
            ret_shape.push_back(shape[i]);
            ret_shape_pad.push_back(shape[i]);
        }
    }
    return std::tuple<Shape, Shape>(std::move(ret_shape),
                                    std::move(ret_shape_pad));
}

template <typename F>
NdArray ReduceAxisAll(const NdArray& src, const float init_v, F op) {
    const float* data = src.data();
    float ret = init_v;
    for (size_t i = 0; i < src.size(); i++) {
        ret = op(ret, *(data++));
    }
    return {ret};
}

template <typename F>
NdArray ReduceAxis(const NdArray& src, const Axes& axes, const float init_v,
                   F op) {
    if (axes.size() == 0) {
        // No Axis -> Reduce all
        return ReduceAxisAll(src, init_v, op);
    } else {
        // Check it is possible to reduce.
        const Shape& src_shape = src.shape();
        const auto& ret_shapes = CheckReductable(src_shape, axes);
        const Shape& ret_shape = std::get<0>(ret_shapes);
        const Shape& ret_shape_pad = std::get<1>(ret_shapes);

        // Pre-compute child sizes
        const auto& ret_child_sizes = ComputeChildSizes(ret_shape_pad);
        const auto& src_child_sizes = ComputeChildSizes(src_shape);

        // Sort axes
        Axes sorted_axes = axes;
        std::sort(sorted_axes.begin(), sorted_axes.end());

        // Result array with value initialization
        NdArray ret(ret_shape, init_v);

        // Reduce
        float* ret_data = ret.data();
        const float* src_data = src.data();
        for (size_t src_idx = 0; src_idx < src.size(); src_idx++) {
            // Result index
            const int ret_idx = ComputeReducedIndex(
                    src_idx, ret_child_sizes, src_child_sizes, sorted_axes);
            // Reduce one source element
            float& ret_v = ret_data[ret_idx];
            ret_v = op(ret_v, *(src_data++));
        }

        return ret;
    }
}

template <typename F>
NdArray ReduceAxisNoEmpty(const NdArray& src, const Axes& axes,
                          const float init_v, F op) {
    // Check empty
    if (src.size() == 0) {
        throw std::runtime_error("zero-size array to reduction operation");
    }
    // Call normally
    return ReduceAxis(src, axes, init_v, op);
}

// ----------------------- Utilities for NdArray (Print) -----------------------
static void OutputArrayLine(std::ostream& os, const float*& data, size_t size) {
    os << "[";  // Begin of a line
    for (size_t i = 0; i < size; i++) {
        os << *(data++);  // Output an element
        if (i == size - 1) {
            os << "]";  // End of a line
        } else {
            os << ", ";  // Splitter of an element
        }
    }
}

static void OutputArrayMultiDim(std::ostream& os, const float*& data,
                                const Shape& shape, size_t depth) {
    for (size_t i = 0; i < static_cast<size_t>(shape[depth]); i++) {
        // Heading
        if (i == 0) {
            os << "[";  // begin of array
        } else {
            for (size_t d = 0; d < depth + 1; d++) {  // array indent
                os << " ";
            }
        }

        // Output internal array
        if (depth == shape.size() - 2) {
            OutputArrayLine(os, data, static_cast<size_t>(shape[depth + 1]));
        } else {
            OutputArrayMultiDim(os, data, shape, depth + 1);
        }

        // Tailing
        if (i == static_cast<size_t>(shape[depth]) - 1) {
            os << "]";  // End of array
        } else {
            os << "," << std::endl;  // Splitter of array
        }
    }
}

static void OutputNdArray(std::ostream& os, const NdArray& x) {
    const size_t size = x.size();
    const Shape& shape = x.shape();
    const float* data = x.data();

    if (size == 0 || shape.size() == 0) {
        // Empty
        os << "[]";
    } else if (shape.size() == 1) {
        // 1-dim
        OutputArrayLine(os, data, size);
    } else {
        // Multi-dim
        OutputArrayMultiDim(os, data, shape, 0);
    }
}

static void OutputShape(std::ostream& os, const Shape& shape) {
    os << "[";
    for (size_t i = 0; i < shape.size(); i++) {
        os << shape[i];
        if (i < shape.size() - 1) {
            os << ", ";
        }
    }
    os << "]";
}

// -------------------- Utilities for NdArray (Dot product) --------------------
static NdArray DotNdArray1d(const NdArray& lhs, const NdArray& rhs) {
    if (lhs.size() != rhs.size()) {
        throw std::runtime_error("Invalid size for inner product of 1D");
    }
    // Inner product of vectors
    const float* l_data = lhs.data();
    const float* r_data = rhs.data();
    float sum = 0.f;
    for (size_t i = 0; i < lhs.size(); i++) {
        sum += l_data[i] * r_data[i];
    }
    return {sum};
}

static NdArray DotNdArray2d(const NdArray& lhs, const NdArray& rhs) {
    const Shape& l_shape = lhs.shape();  // 2 == size
    const Shape& r_shape = rhs.shape();  // 2 == size
    if (l_shape[1] != r_shape[0]) {
        throw std::runtime_error("Invalid size for inner product of 2D");
    }
    // Inner product of 2D matrix
    const int n_row = l_shape[0];
    const int n_col = r_shape[1];
    const int n_contract = l_shape[1];
    const int& n_l_col = n_contract;
    const int& n_r_col = n_col;
    NdArray ret({n_row, n_col});
    float* ret_data = ret.data();
    const float* l_data = lhs.data();
    const float* r_data = rhs.data();
    for (int row_idx = 0; row_idx < n_row; row_idx++) {
        for (int col_idx = 0; col_idx < n_col; col_idx++) {
            float sum = 0.f;
            for (int i = 0; i < n_contract; i++) {
                sum += l_data[row_idx * n_l_col + i] *
                       r_data[i * n_r_col + col_idx];
            }
            *(ret_data++) = sum;
        }
    }
    return ret;
}

static NdArray DotNdArrayNdMd(const NdArray& lhs, const NdArray& rhs) {
    const Shape& l_shape = lhs.shape();  // 1 <= size
    const Shape& r_shape = rhs.shape();  // 2 <= size

    // The last axis of left and the second-to-last axis of right must be same.
    if (l_shape[l_shape.size() - 1] != r_shape[r_shape.size() - 2]) {
        throw std::runtime_error("Invalid shape for dot product");
    }

    // Result shape
    Shape ret_shape(l_shape.begin(), l_shape.end() - 1);
    ret_shape.insert(ret_shape.end(), r_shape.begin(), r_shape.end() - 2);
    ret_shape.push_back(r_shape.back());

    // Compute child sizes
    //   [2, 3, (4)] [5, 6, (4), 7] -> [2, 3, 5, 6, 7]
    const int ret_child_size_2 = std::accumulate(
            ret_shape.begin() + static_cast<long>(l_shape.size()) - 1,
            ret_shape.end(), 1, std::multiplies<int>());  // [(5), (6), 4, (7)]
    const int ret_child_size_1 = ret_shape.back();        // [5, 6, 4, (7)]
    const int l_child_size = l_shape.back();              // [2, 3, (4)]
    const int r_child_size_2 =
            r_shape.end()[-1] * r_shape.end()[-2];  // [5, 6, (4), (7)]
    const int r_child_size_1 = ret_child_size_1;    // [2, 3, (4)]

    // Basic matrix product
    NdArray ret(ret_shape);
    const int n_ret = static_cast<int>(ret.size());
    const int n_contract = l_child_size;
    float* ret_data = ret.data();
    const float* l_data = lhs.data();
    const float* r_data = rhs.data();
    for (int ret_idx = 0; ret_idx < n_ret; ret_idx++) {
        // Compute left/right index from `ret_idx`
        const int l_idx = (ret_idx / ret_child_size_2) * l_child_size;
        const int r_idx_2 = (ret_idx % ret_child_size_2) / ret_child_size_1;
        const int r_idx_1 = ret_idx % ret_child_size_1;
        const int r_idx = (r_idx_2 * r_child_size_2) + r_idx_1;
        // Sum up
        float sum = 0.f;
        for (int i = 0; i < n_contract; i++) {
            sum += l_data[l_idx + i] * r_data[r_idx + (i * r_child_size_1)];
        }
        // Register
        ret_data[ret_idx] = sum;
    }
    return ret;
}

// ------------------- Utilities for NdArray (Cross product) -------------------
static void CrossNdArray1d1dShape33(float* ret_data, const float* l_data,
                                    const float* r_data) {
    // lhs.shape() == {3} && rhs.shape == {3}
    *(ret_data++) = l_data[1] * r_data[2] - l_data[2] * r_data[1];
    *(ret_data++) = l_data[2] * r_data[0] - l_data[0] * r_data[2];
    *ret_data = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

static void CrossNdArray1d1dShape32(float* ret_data, const float* l_data,
                                    const float* r_data) {
    // lhs.shape() == {3} && rhs.shape == {2}
    *(ret_data++) = -l_data[2] * r_data[1];
    *(ret_data++) = l_data[2] * r_data[0];
    *ret_data = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

static void CrossNdArray1d1dShape23(float* ret_data, const float* l_data,
                                    const float* r_data) {
    // lhs.shape() == {3} && rhs.shape == {3}
    *(ret_data++) = l_data[1] * r_data[2];
    *(ret_data++) = -l_data[0] * r_data[2];
    *ret_data = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

static void CrossNdArray1d1dShape22(float* ret_data, const float* l_data,
                                    const float* r_data) {
    // lhs.shape() == {2} && rhs.shape == {2}
    *ret_data = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

template <typename F>
NdArray CrossNdArrayNdMd(const NdArray& lhs, const NdArray& rhs,
                         size_t last_size, F op) {
    const Shape& l_shape = lhs.shape();
    const Shape& r_shape = rhs.shape();
    Shape ret_shape = CheckBroadcastable({l_shape.begin(), l_shape.end() - 1},
                                         {r_shape.begin(), r_shape.end() - 1});
    ret_shape.push_back(last_size);
    // Apply broadcast
    NdArray ret(ret_shape);
    ApplyOpBroadcast(ret, lhs, rhs, 1, op);
    return ret;
}

// ---------------------- Utilities for NdArray (Inverse) ----------------------
static int CheckInversable(const Shape& shape) {
    if (shape.size() < 2) {
        throw std::runtime_error("For matrix inverse, require at least 2-dim");
    }
    const int size = shape.back();
    if (size != shape.end()[-2]) {
        throw std::runtime_error(
                "For matrix inverse, last 2 dimensions of the"
                " array must be square");
    }
    return size;
}

static void InvertNdArray2d(float* ret_data, const float* src_data, int order) {
    const int order_2 = order * 2;
    std::unique_ptr<float[]> tmp(new float[order_2 * order_2]);
    float* tmp_data = tmp.get();

    for (int row = 0; row < order; row++) {
        for (int col = 0; col < order; col++) {
            tmp_data[row * order_2 + col] = src_data[row * order + col];
        }
    }
    for (int row = 0; row < order; row++) {
        for (int col = order; col < order_2; col++) {
            tmp_data[row * order_2 + col] = (row == col - order) ? 1.f : 0.f;
        }
    }
    for (int row = 0; row < order; row++) {
        float t = tmp_data[row * order_2 + row];
        if (std::abs(t) == 0.f) {
            t = 0.00001f;  // Escape zero (TODO: More pricise method)
        }
        for (int col = row; col < order_2; col++) {
            tmp_data[row * order_2 + col] /= t;
        }
        for (int col = 0; col < order; col++) {
            if (row == col) {
                continue;
            }
            t = tmp_data[col * order_2 + row];
            for (int k = 0; k < order_2; k++) {
                tmp_data[col * order_2 + k] -= t * tmp_data[row * order_2 + k];
            }
        }
    }

    for (int row = 0; row < order; row++) {
        for (int col = order; col < order_2; col++) {
            *(ret_data++) = tmp_data[row * order_2 + col];
        }
    }
}

static void InvertNdArrayNd(float* ret_data, const float* src_data,
                            const Shape& src_shape, size_t src_size) {
    // Check it is possible to invert
    const int order = CheckInversable(src_shape);
    // Compute invese for each lower 2 dimension.
    const int one_size = order * order;
    const int n = static_cast<int>(src_size) / one_size;
    for (int i = 0; i < n; i++) {
        InvertNdArray2d(ret_data, src_data, order);
        ret_data += one_size;
        src_data += one_size;
    }
}

static NdArray InvertNdArray(const NdArray& src) {
    // Create new array
    NdArray ret(src.shape());
    // Compute inverse
    InvertNdArrayNd(ret.data(), src.data(), src.shape(), src.size());
    return ret;
}

static NdArray InvertNdArrayInplace(NdArray&& src) {
    // Compute inverse
    InvertNdArrayNd(src.data(), src.data(), src.shape(), src.size());
    return src;
}

// =============================================================================
// ============================ NdArray Definition =============================
// =============================================================================
NdArray::NdArray() : m_sub(std::make_shared<Substance>()) {}

NdArray::NdArray(std::shared_ptr<Substance> sub) : m_sub(sub) {}

NdArray::NdArray(const NdArray& lhs) = default;  // shallow copy

NdArray::NdArray(NdArray&& lhs) noexcept
    : m_sub(std::move(lhs.m_sub)) {}  // move

NdArray& NdArray::operator=(const NdArray& lhs) = default;  // shallow copy

NdArray& NdArray::operator=(NdArray&& lhs) {  // move
    m_sub = std::move(lhs.m_sub);
    return *this;
}

NdArray::~NdArray() = default;

// --------------------------------- Substance ---------------------------------
class NdArray::Substance {
public:
    Substance() {}
    Substance(size_t size_, const Shape& shape_)
        : size(size_),
          shape(shape_),
          v(new float[size_], std::default_delete<float[]>()) {}
    size_t size = 0;
    Shape shape = {0};
    std::shared_ptr<float> v;  // C++17: Replace with `shared_ptr<float[]>`.
};

// ------------------------------- Static Member -------------------------------
std::random_device NdArray::s_rand_seed;
std::mt19937 NdArray::s_rand_engine(s_rand_seed());

// -------------------- Constructors with Float Initializers -------------------
NdArray::NdArray(FloatList<0> init_list) : NdArray(CheckFListShape(init_list)) {
    // Fill after empty initialization
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<1> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<2> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<3> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<4> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<5> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<6> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<7> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

NdArray::NdArray(FloatList<9> init_list) : NdArray(CheckFListShape(init_list)) {
    CopyFListElems(init_list, m_sub->v.get());
}

// -------------------------- Constructors with Shape --------------------------
NdArray::NdArray(const InitShape& shape) : NdArray(Shape(shape)) {
    // Just pass initializer list to `Shape` (== std::vector<int>).
}

NdArray::NdArray(const Shape& shape) {
    // Compute total size
    size_t size = 1;
    for (auto&& s : shape) {
        if (s < 0) {
            throw std::runtime_error("Invalid shape format (neg)");
        }
        size *= static_cast<size_t>(s);
    }
    // Create substance
    m_sub = std::make_shared<Substance>(size, shape);
}

NdArray::NdArray(const Shape& shape, float fill_v) : NdArray(shape) {
    // Fill after empty initialization
    fill(fill_v);
}

// ------------------------------- Static Methods ------------------------------
NdArray NdArray::Empty(const Shape& shape) {
    return NdArray(shape);
}

NdArray NdArray::Zeros(const Shape& shape) {
    return NdArray(shape, 0.f);
}

NdArray NdArray::Ones(const Shape& shape) {
    return NdArray(shape, 1.f);
}

template <typename... S>
NdArray NdArray::Empty(S... shape) {
    return Empty({shape...});  // Unpack
}

template <typename... S>
NdArray NdArray::Zeros(S... shape) {
    return Zeros({shape...});  // Unpack
}

template <typename... S>
NdArray NdArray::Ones(S... shape) {
    return Ones({shape...});  // Unpack
}

NdArray NdArray::Arange(float stop) {
    return Arange(0.f, stop, 1.f);
}

NdArray NdArray::Arange(float start, float stop, float step) {
    const size_t n = static_cast<size_t>(std::ceil((stop - start) / step));
    // Create empty array
    NdArray ret({static_cast<int>(n)});
    // Fill by step
    float* data = ret.data();
    for (size_t i = 0; i < n; i++) {
        data[i] = start + step * static_cast<float>(i);
    }
    return ret;
}

void NdArray::Seed() {
    s_rand_engine = std::mt19937(s_rand_seed());
}

void NdArray::Seed(uint32_t seed) {
    s_rand_engine = std::mt19937(seed);
}

NdArray NdArray::Uniform(float low, float high, const Shape& shape) {
    // Create uniform distribution
    std::uniform_real_distribution<> dist(low, high);
    // Create random array
    return CreateRandomArray(shape, dist, s_rand_engine);
}

NdArray NdArray::Uniform(const Shape& shape) {
    return Uniform(0.f, 1.f, shape);
}

NdArray NdArray::Normal(float loc, float scale, const Shape& shape) {
    // Create normal distribution
    std::normal_distribution<> dist(loc, scale);
    // Create random array
    return CreateRandomArray(shape, dist, s_rand_engine);
}

NdArray NdArray::Normal(const Shape& shape) {
    return Normal(0.f, 1.f, shape);
}

// ------------------------------- Basic Methods -------------------------------
uintptr_t NdArray::id() const {
    return reinterpret_cast<uintptr_t>(m_sub->v.get());  // pointer of array
}

bool NdArray::empty() const {
    return m_sub->size == 0;
}

size_t NdArray::size() const {
    return m_sub->size;
}

const Shape& NdArray::shape() const {
    return m_sub->shape;
}

size_t NdArray::ndim() const {
    return m_sub->shape.size();
}

float* NdArray::data() {
    return m_sub->v.get();
}

const float* NdArray::data() const {
    return m_sub->v.get();
}

void NdArray::fill(float v) {
    std::fill_n(m_sub->v.get(), m_sub->size, v);
}

NdArray NdArray::copy() const {
    // Create new substance
    auto sub = std::make_shared<Substance>(m_sub->size, m_sub->shape);
    // Copy array data
    float* dst_data = sub->v.get();
    const float* src_data = m_sub->v.get();
    const int size = m_sub->size;
    for (int i = 0; i < size; i++) {
        *(dst_data++) = *(src_data++);
    }
    // Create new array
    return NdArray(sub);
}

// ----------------------------- Begin/End Methods -----------------------------
float* NdArray::begin() {
    return m_sub->v.get();
}

float* NdArray::end() {
    return m_sub->v.get() + m_sub->size;
}

const float* NdArray::begin() const {
    return m_sub->v.get();
}

const float* NdArray::end() const {
    return m_sub->v.get() + m_sub->size;
}

// ------------------------------- Cast Operator -------------------------------
NdArray::operator float() const {
    if (m_sub->size != 1) {
        throw std::runtime_error("Only size-1 arrays can be casted to float");
    }
    return *(m_sub->v.get());
}

// ------------------------------- Index Methods -------------------------------
float& NdArray::operator[](int i) {
    // Use the same implementation of constant method.
    return const_cast<float&>(static_cast<const NdArray&>(*this)[i]);
}

const float& NdArray::operator[](int i) const {
    // Make the index positive
    const int p_idx = (0 <= i) ? i : static_cast<int>(m_sub->size) + i;
    // Direct access with range check
    return *(m_sub->v.get() + p_idx);
}

float& NdArray::operator[](const Index& index) {
    // Use the same implementation of constant method.
    return const_cast<float&>(static_cast<const NdArray&>(*this)[index]);
}

const float& NdArray::operator[](const Index& index) const {
    const auto& shape = m_sub->shape;
    if (index.size() != shape.size()) {
        throw std::runtime_error("Invalid index size");
    }
    // Compute flatten index
    int i = 0;
    for (size_t d = 0; d < index.size(); d++) {
        // Compute `i = i * shape + index` recurrently
        i *= shape[d];
        // Make the index positive
        const int p_idx = (0 <= index[d]) ? index[d] : shape[d] + index[d];
        i += p_idx;
    }
    // Direct access
    return *(m_sub->v.get() + i);
}

template <typename... I>
float& NdArray::operator()(I... index) {
    // Use the same implementation of constant method.
    return const_cast<float&>(static_cast<const NdArray&>(*this)(index...));
}

template <typename... I>
const float& NdArray::operator()(I... index) const {
    // Pass to operator[]
    return (*this)[{index...}];
}

// ------------------------------- Reshape Method ------------------------------
NdArray NdArray::reshape(const Shape& shape) const {
    // Check shape validity
    size_t unknown_idx = shape.size();
    size_t size = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] < 0) {
            if (unknown_idx != shape.size()) {
                throw std::runtime_error("Invalid shape format (multi-neg)");
            } else {
                unknown_idx = i;
            }
        } else {
            size *= static_cast<size_t>(shape[i]);
        }
    }
    Shape new_shape = shape;
    if (unknown_idx == shape.size()) {
        if (m_sub->size != size) {
            std::stringstream ss;
            ss << "Invalid reshape (" << m_sub->size << "->" << size << ")";
            throw std::runtime_error(ss.str());
        }
    } else {
        if (m_sub->size % size != 0) {
            throw std::runtime_error("Invalid reshape (-1)");
        }
        new_shape[unknown_idx] = static_cast<int>(m_sub->size / size);
    }

    // Create reshaped array
    NdArray ret;
    ret.m_sub->size = m_sub->size;            // Same size
    ret.m_sub->shape = std::move(new_shape);  // New shape
    ret.m_sub->v = m_sub->v;                  // Shared elements
    return ret;
}

template <typename... S>
NdArray NdArray::reshape(S... shape) const {
    // Pass to `reshape(Shape)`
    return reshape({shape...});
}

NdArray NdArray::flatten() const {
    return reshape({-1}).copy();
}

NdArray NdArray::ravel() const {
    return reshape({-1});
}

// -------------------------------- Slice Method -------------------------------
NdArray NdArray::slice(const SliceIndex& slice_index) const {
    const Shape& shape = m_sub->shape;

    // Compute slice shape and new positive index
    Shape slice_shape;
    SliceIndex new_index;
    for (size_t i = 0; i < shape.size(); i++) {
        const auto& si = slice_index[i];
        if (slice_index.size() <= i) {
            // All
            slice_shape.push_back(shape[i]);
            new_index.push_back({0, shape[i]});
        } else {
            // Make index positive
            int s = (0 <= si.first) ? si.first : shape[i] + si.first;
            int e = (0 <= si.second) ? si.second : shape[i] + si.second;
            // Clamp
            s = Clamp(s, 0, shape[i] - 1);  // Start must be in range.
            e = Clamp(e, 0, shape[i]);      // End can be next of the last.
            // Register
            slice_shape.push_back(e - s);
            new_index.push_back({s, e});
        }
    }

    // Copy to slice array
    return CopySlice((*this), slice_shape, new_index);
}

template <typename... I>
NdArray NdArray::slice(std::initializer_list<I>... slice_index) const {
    // Cast `initializer_list` to `pair`, and pass to 'slice(SliceIndex)'
    return slice(SliceIndex{CvtToSliceIndexItem(slice_index)...});
}

// --------------------------------- Dot Method --------------------------------
NdArray NdArray::dot(const NdArray& other) const {
    const NdArray& lhs = *this;
    const NdArray& rhs = other;
    const Shape& l_shape = lhs.shape();
    const Shape& r_shape = rhs.shape();
    if (lhs.size() == 0 || rhs.size() == 0) {
        // Empty array
        throw std::runtime_error("Dot product of empty array");
    } else if (lhs.size() == 1) {
        // Simple multiply (left)
        return static_cast<float>(lhs) * rhs;
    } else if (rhs.size() == 1) {
        // Simple multiply (right)
        return lhs * static_cast<float>(rhs);
    } else if (l_shape.size() == 1 && r_shape.size() == 1) {
        // Inner product of vector (1D, 1D)
        return DotNdArray1d(lhs, rhs);
    } else if (l_shape.size() == 2 && r_shape.size() == 2) {
        // Inner product of 2D matrix (2D, 2D)
        // Special version of NDMD. This is for faster calculation.
        return DotNdArray2d(lhs, rhs);
    } else if (l_shape.size() == 2 && r_shape.size() == 1) {
        // Inner product of 2D matrix and vector (2D, 1D)
        // Special version of ND1D. This is for faster calculation.
        const int n_elem = l_shape[0];
        return DotNdArray2d(lhs, rhs.reshape(r_shape[0], 1)).reshape(n_elem);
    } else if (r_shape.size() == 1) {
        // Broadcast right 1D array
        const Shape shape(l_shape.begin(), l_shape.end() - 1);
        return DotNdArrayNdMd(lhs, rhs.reshape(r_shape[0], 1)).reshape(shape);
    } else {
        // Basic matrix product
        return DotNdArrayNdMd(lhs, rhs);
    }
}

NdArray NdArray::dot(float other) const {
    // Simple multiply (right)
    return (*this) * other;
}

// -------------------------------- Cross Method -------------------------------
NdArray NdArray::cross(const NdArray& other) const {
    const NdArray& lhs = *this;
    const NdArray& rhs = other;
    if (lhs.size() == 0 || rhs.size() == 0) {
        // Empty array
        throw std::runtime_error("Cross product of empty array");
    }
    const Shape& l_shape = lhs.shape();
    const Shape& r_shape = rhs.shape();
    const int l_back = l_shape.back();
    const int r_back = r_shape.back();
    if (l_shape.size() == 1 && r_shape.size() == 1) {
        // 1D cross
        if (l_back == 3 && r_back == 3) {  // [3].cross([3]) -> [3]
            NdArray ret({3});
            CrossNdArray1d1dShape33(ret.data(), lhs.data(), rhs.data());
            return ret;
        } else if (l_back == 3 && r_back == 2) {  // [3].cross([2]) -> [3]
            NdArray ret({3});
            CrossNdArray1d1dShape32(ret.data(), lhs.data(), rhs.data());
            return ret;
        } else if (l_back == 2 && r_back == 3) {  // [2].cross([3]) -> [3]
            NdArray ret({3});
            CrossNdArray1d1dShape23(ret.data(), lhs.data(), rhs.data());
            return ret;
        } else if (l_back == 2 && r_back == 2) {  // [2].cross([2]) -> [1]
            NdArray ret({1});
            CrossNdArray1d1dShape22(ret.data(), lhs.data(), rhs.data());
            return ret;
        }
    } else {
        // ND cross
        if (l_back == 3 && r_back == 3) {  // [3].cross([3]) -> [3]
            return CrossNdArrayNdMd(lhs, rhs, 3, CrossNdArray1d1dShape33);
        } else if (l_back == 3 && r_back == 2) {  // [2].cross([3]) -> [3]
            return CrossNdArrayNdMd(lhs, rhs, 3, CrossNdArray1d1dShape32);
        } else if (l_back == 2 && r_back == 3) {  // [3].cross([2]) -> [3]
            return CrossNdArrayNdMd(lhs, rhs, 3, CrossNdArray1d1dShape23);
        } else if (l_back == 2 && r_back == 2) {  // [2].cross([2]) -> [1]
            auto&& ret = CrossNdArrayNdMd(lhs, rhs, 1, CrossNdArray1d1dShape22);
            const Shape& ret_shape = ret.shape();  // Remove last dim
            return ret.reshape(Shape{ret_shape.begin(), ret_shape.end() - 1});
        }
    }
    throw std::runtime_error(
            "incompatible dimensions for cross product"
            " (dimension must be 2 or 3)");
}

// -------------------------------- Axis Method --------------------------------
NdArray NdArray::sum(const Axes& axes) const {
    return Sum(*this, axes);
}

NdArray NdArray::min(const Axes& axes) const {
    return Min(*this, axes);
}

NdArray NdArray::max(const Axes& axes) const {
    return Max(*this, axes);
}

NdArray NdArray::mean(const Axes& axes) const {
    return Mean(*this, axes);
}

// ---------------------- Template Method Specializations ----------------------
// Assuming up to 10 dimensions.
// For `Empty(S... shape)`
template NdArray NdArray::Empty(int);
template NdArray NdArray::Empty(int, int);
template NdArray NdArray::Empty(int, int, int);
template NdArray NdArray::Empty(int, int, int, int);
template NdArray NdArray::Empty(int, int, int, int, int);
template NdArray NdArray::Empty(int, int, int, int, int, int);
template NdArray NdArray::Empty(int, int, int, int, int, int, int);
template NdArray NdArray::Empty(int, int, int, int, int, int, int, int);
template NdArray NdArray::Empty(int, int, int, int, int, int, int, int, int);
template NdArray NdArray::Empty(int, int, int, int, int, int, int, int, int,
                                int);
// For `Zeros(S... shape)`
template NdArray NdArray::Zeros(int);
template NdArray NdArray::Zeros(int, int);
template NdArray NdArray::Zeros(int, int, int);
template NdArray NdArray::Zeros(int, int, int, int);
template NdArray NdArray::Zeros(int, int, int, int, int);
template NdArray NdArray::Zeros(int, int, int, int, int, int);
template NdArray NdArray::Zeros(int, int, int, int, int, int, int);
template NdArray NdArray::Zeros(int, int, int, int, int, int, int, int);
template NdArray NdArray::Zeros(int, int, int, int, int, int, int, int, int);
template NdArray NdArray::Zeros(int, int, int, int, int, int, int, int, int,
                                int);
// For `Ones(S... shape)`
template NdArray NdArray::Ones(int);
template NdArray NdArray::Ones(int, int);
template NdArray NdArray::Ones(int, int, int);
template NdArray NdArray::Ones(int, int, int, int);
template NdArray NdArray::Ones(int, int, int, int, int);
template NdArray NdArray::Ones(int, int, int, int, int, int);
template NdArray NdArray::Ones(int, int, int, int, int, int, int);
template NdArray NdArray::Ones(int, int, int, int, int, int, int, int);
template NdArray NdArray::Ones(int, int, int, int, int, int, int, int, int);
template NdArray NdArray::Ones(int, int, int, int, int, int, int, int, int,
                               int);
// For `float& operator()(I... index)`
template float& NdArray::operator()(int);
template float& NdArray::operator()(int, int);
template float& NdArray::operator()(int, int, int);
template float& NdArray::operator()(int, int, int, int);
template float& NdArray::operator()(int, int, int, int, int);
template float& NdArray::operator()(int, int, int, int, int, int);
template float& NdArray::operator()(int, int, int, int, int, int, int);
template float& NdArray::operator()(int, int, int, int, int, int, int, int);
template float& NdArray::operator()(int, int, int, int, int, int, int, int,
                                    int);
template float& NdArray::operator()(int, int, int, int, int, int, int, int, int,
                                    int);
// For `const float& operator()(I... index) const`
template const float& NdArray::operator()(int) const;
template const float& NdArray::operator()(int, int) const;
template const float& NdArray::operator()(int, int, int) const;
template const float& NdArray::operator()(int, int, int, int) const;
template const float& NdArray::operator()(int, int, int, int, int) const;
template const float& NdArray::operator()(int, int, int, int, int, int) const;
template const float& NdArray::operator()(int, int, int, int, int, int,
                                          int) const;
template const float& NdArray::operator()(int, int, int, int, int, int, int,
                                          int) const;
template const float& NdArray::operator()(int, int, int, int, int, int, int,
                                          int, int) const;
template const float& NdArray::operator()(int, int, int, int, int, int, int,
                                          int, int, int) const;
// For `NdArray reshape(S... shape) const`
template NdArray NdArray::reshape(int) const;
template NdArray NdArray::reshape(int, int) const;
template NdArray NdArray::reshape(int, int, int) const;
template NdArray NdArray::reshape(int, int, int, int) const;
template NdArray NdArray::reshape(int, int, int, int, int) const;
template NdArray NdArray::reshape(int, int, int, int, int, int) const;
template NdArray NdArray::reshape(int, int, int, int, int, int, int) const;
template NdArray NdArray::reshape(int, int, int, int, int, int, int, int) const;
template NdArray NdArray::reshape(int, int, int, int, int, int, int, int,
                                  int) const;
template NdArray NdArray::reshape(int, int, int, int, int, int, int, int, int,
                                  int) const;
template NdArray NdArray::reshape(int, int, int, int, int, int, int, int, int,
                                  int, int) const;
// For `NdArray slice(std::initializer_list<I>... slice_index) const`
using ISII = std::initializer_list<int>;  // Initializer of Slice Index Item
template NdArray NdArray::slice(ISII) const;
template NdArray NdArray::slice(ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII, ISII,
                                ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII, ISII, ISII,
                                ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII, ISII, ISII,
                                ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII, ISII, ISII,
                                ISII, ISII, ISII) const;

// --------------------------------- Operators ---------------------------------
// Print
std::ostream& operator<<(std::ostream& os, const NdArray& x) {
    OutputNdArray(os, x);
    return os;
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
    OutputShape(os, shape);
    return os;
}

// Single
NdArray operator+(const NdArray& x) {
    return x.copy();  // Numpy behavior
}

NdArray operator-(const NdArray& x) {
    return ApplySingleOp(x, [](float v) { return -v; });
}

// Arithmetic (NdArray, NdArray)
NdArray operator+(const NdArray& lhs, const NdArray& rhs) {
    return Add(lhs, rhs);
}

NdArray operator-(const NdArray& lhs, const NdArray& rhs) {
    return Subtract(lhs, rhs);
}

NdArray operator*(const NdArray& lhs, const NdArray& rhs) {
    return Multiply(lhs, rhs);
}

NdArray operator/(const NdArray& lhs, const NdArray& rhs) {
    return Divide(lhs, rhs);
}

// Arithmetic (NdArray, float)
NdArray operator+(const NdArray& lhs, const float& rhs) {
    return Add(lhs, rhs);
}

NdArray operator-(const NdArray& lhs, const float& rhs) {
    return Subtract(lhs, rhs);
}

NdArray operator*(const NdArray& lhs, const float& rhs) {
    return Multiply(lhs, rhs);
}

NdArray operator/(const NdArray& lhs, const float& rhs) {
    return Divide(lhs, rhs);
}

// Arithmetic (float, NdArray)
NdArray operator+(const float& lhs, const NdArray& rhs) {
    return Add(lhs, rhs);
}

NdArray operator-(const float& lhs, const NdArray& rhs) {
    return Subtract(lhs, rhs);
}

NdArray operator*(const float& lhs, const NdArray& rhs) {
    return Multiply(lhs, rhs);
}

NdArray operator/(const float& lhs, const NdArray& rhs) {
    return Divide(lhs, rhs);
}

// Comparison (NdArray, NdArray)
NdArray operator==(const NdArray& lhs, const NdArray& rhs) {
    return Equal(lhs, rhs);
}

NdArray operator!=(const NdArray& lhs, const NdArray& rhs) {
    return NotEqual(lhs, rhs);
}

NdArray operator>(const NdArray& lhs, const NdArray& rhs) {
    return Greater(lhs, rhs);
}

NdArray operator>=(const NdArray& lhs, const NdArray& rhs) {
    return GreaterEqual(lhs, rhs);
}

NdArray operator<(const NdArray& lhs, const NdArray& rhs) {
    return Less(lhs, rhs);
}

NdArray operator<=(const NdArray& lhs, const NdArray& rhs) {
    return LessEqual(lhs, rhs);
}

// Comparison (NdArray, float)
NdArray operator==(const NdArray& lhs, float rhs) {
    return Equal(lhs, rhs);
}

NdArray operator!=(const NdArray& lhs, float rhs) {
    return NotEqual(lhs, rhs);
}

NdArray operator>(const NdArray& lhs, float rhs) {
    return Greater(lhs, rhs);
}

NdArray operator>=(const NdArray& lhs, float rhs) {
    return GreaterEqual(lhs, rhs);
}

NdArray operator<(const NdArray& lhs, float rhs) {
    return Less(lhs, rhs);
}

NdArray operator<=(const NdArray& lhs, float rhs) {
    return LessEqual(lhs, rhs);
}

// Comparison (float, NdArray)
NdArray operator==(float lhs, const NdArray& rhs) {
    return Equal(lhs, rhs);
}

NdArray operator!=(float lhs, const NdArray& rhs) {
    return NotEqual(lhs, rhs);
}

NdArray operator>(float lhs, const NdArray& rhs) {
    return Greater(lhs, rhs);
}

NdArray operator>=(float lhs, const NdArray& rhs) {
    return GreaterEqual(lhs, rhs);
}

NdArray operator<(float lhs, const NdArray& rhs) {
    return Less(lhs, rhs);
}

NdArray operator<=(float lhs, const NdArray& rhs) {
    return LessEqual(lhs, rhs);
}

// ----------------------------- In-place Operators ----------------------------
// Single
NdArray operator+(NdArray&& x) {
    return std::move(x);
}

NdArray operator-(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x), [](float v) { return -v; });
}

// Arithmetic (NdArray, NdArray)
NdArray operator+(NdArray&& lhs, NdArray&& rhs) {
    return Add(std::move(lhs), std::move(rhs));
}

NdArray operator+(const NdArray& lhs, NdArray&& rhs) {
    return Add(lhs, std::move(rhs));
}

NdArray operator+(NdArray&& lhs, const NdArray& rhs) {
    return Add(std::move(lhs), rhs);
}

NdArray operator-(NdArray&& lhs, NdArray&& rhs) {
    return Subtract(std::move(lhs), std::move(rhs));
}

NdArray operator-(const NdArray& lhs, NdArray&& rhs) {
    return Subtract(lhs, std::move(rhs));
}

NdArray operator-(NdArray&& lhs, const NdArray& rhs) {
    return Subtract(std::move(lhs), rhs);
}

NdArray operator*(NdArray&& lhs, NdArray&& rhs) {
    return Multiply(std::move(lhs), std::move(rhs));
}

NdArray operator*(const NdArray& lhs, NdArray&& rhs) {
    return Multiply(lhs, std::move(rhs));
}

NdArray operator*(NdArray&& lhs, const NdArray& rhs) {
    return Multiply(std::move(lhs), rhs);
}

NdArray operator/(NdArray&& lhs, NdArray&& rhs) {
    return Divide(std::move(lhs), std::move(rhs));
}

NdArray operator/(const NdArray& lhs, NdArray&& rhs) {
    return Divide(lhs, std::move(rhs));
}

NdArray operator/(NdArray&& lhs, const NdArray& rhs) {
    return Divide(std::move(lhs), rhs);
}

// Arithmetic (NdArray, float)
NdArray operator+(NdArray&& lhs, float rhs) {
    return Add(std::move(lhs), rhs);
}

NdArray operator-(NdArray&& lhs, float rhs) {
    return Subtract(std::move(lhs), rhs);
}

NdArray operator*(NdArray&& lhs, float rhs) {
    return Multiply(std::move(lhs), rhs);
}

NdArray operator/(NdArray&& lhs, float rhs) {
    return Divide(std::move(lhs), rhs);
}

// Arithmetic (float, NdArray)
NdArray operator+(float lhs, NdArray&& rhs) {
    return Add(lhs, std::move(rhs));
}

NdArray operator-(float lhs, NdArray&& rhs) {
    return Subtract(lhs, std::move(rhs));
}

NdArray operator*(float lhs, NdArray&& rhs) {
    return Multiply(lhs, std::move(rhs));
}

NdArray operator/(float lhs, NdArray&& rhs) {
    return Divide(lhs, std::move(rhs));
}

// Comparison (NdArray, NdArray)
NdArray operator==(NdArray&& lhs, NdArray&& rhs) {
    return Equal(std::move(lhs), std::move(rhs));
}

NdArray operator==(const NdArray& lhs, NdArray&& rhs) {
    return Equal(lhs, std::move(rhs));
}

NdArray operator==(NdArray&& lhs, const NdArray& rhs) {
    return Equal(std::move(lhs), rhs);
}

NdArray operator!=(NdArray&& lhs, NdArray&& rhs) {
    return NotEqual(std::move(lhs), std::move(rhs));
}

NdArray operator!=(const NdArray& lhs, NdArray&& rhs) {
    return NotEqual(lhs, std::move(rhs));
}

NdArray operator!=(NdArray&& lhs, const NdArray& rhs) {
    return NotEqual(std::move(lhs), rhs);
}

NdArray operator>(NdArray&& lhs, NdArray&& rhs) {
    return Greater(std::move(lhs), std::move(rhs));
}

NdArray operator>(const NdArray& lhs, NdArray&& rhs) {
    return Greater(lhs, std::move(rhs));
}

NdArray operator>(NdArray&& lhs, const NdArray& rhs) {
    return Greater(std::move(lhs), rhs);
}

NdArray operator>=(NdArray&& lhs, NdArray&& rhs) {
    return GreaterEqual(std::move(lhs), std::move(rhs));
}

NdArray operator>=(const NdArray& lhs, NdArray&& rhs) {
    return GreaterEqual(lhs, std::move(rhs));
}

NdArray operator>=(NdArray&& lhs, const NdArray& rhs) {
    return GreaterEqual(std::move(lhs), rhs);
}

NdArray operator<(NdArray&& lhs, NdArray&& rhs) {
    return Less(std::move(lhs), std::move(rhs));
}

NdArray operator<(const NdArray& lhs, NdArray&& rhs) {
    return Less(lhs, std::move(rhs));
}

NdArray operator<(NdArray&& lhs, const NdArray& rhs) {
    return Less(std::move(lhs), rhs);
}

NdArray operator<=(NdArray&& lhs, NdArray&& rhs) {
    return LessEqual(std::move(lhs), std::move(rhs));
}

NdArray operator<=(const NdArray& lhs, NdArray&& rhs) {
    return LessEqual(lhs, std::move(rhs));
}

NdArray operator<=(NdArray&& lhs, const NdArray& rhs) {
    return LessEqual(std::move(lhs), rhs);
}

// Comparison (NdArray, float)
NdArray operator==(NdArray&& lhs, float rhs) {
    return Equal(std::move(lhs), rhs);
}

NdArray operator!=(NdArray&& lhs, float rhs) {
    return NotEqual(std::move(lhs), rhs);
}

NdArray operator>(NdArray&& lhs, float rhs) {
    return Greater(std::move(lhs), rhs);
}

NdArray operator>=(NdArray&& lhs, float rhs) {
    return GreaterEqual(std::move(lhs), rhs);
}

NdArray operator<(NdArray&& lhs, float rhs) {
    return Less(std::move(lhs), rhs);
}

NdArray operator<=(NdArray&& lhs, float rhs) {
    return LessEqual(std::move(lhs), rhs);
}

// Comparison (float, NdArray)
NdArray operator==(float lhs, NdArray&& rhs) {
    return Equal(lhs, std::move(rhs));
}

NdArray operator!=(float lhs, NdArray&& rhs) {
    return NotEqual(lhs, std::move(rhs));
}

NdArray operator>(float lhs, NdArray&& rhs) {
    return Greater(lhs, std::move(rhs));
}

NdArray operator>=(float lhs, NdArray&& rhs) {
    return GreaterEqual(lhs, std::move(rhs));
}

NdArray operator<(float lhs, NdArray&& rhs) {
    return Less(lhs, std::move(rhs));
}

NdArray operator<=(float lhs, NdArray&& rhs) {
    return LessEqual(lhs, std::move(rhs));
}

// Compound Assignment (NdArray, NdArray)
NdArray operator+=(NdArray& lhs, const NdArray& rhs) {
    return lhs = ApplyElemWiseOpInplace(std::move(lhs), rhs, std::plus<float>(),
                                        false);  // force in-place
}

NdArray operator-=(NdArray& lhs, const NdArray& rhs) {
    return lhs = ApplyElemWiseOpInplace(std::move(lhs), rhs,
                                        std::minus<float>(),
                                        false);  // force in-place
}

NdArray operator*=(NdArray& lhs, const NdArray& rhs) {
    return lhs = ApplyElemWiseOpInplace(std::move(lhs), rhs,
                                        std::multiplies<float>(),
                                        false);  // force in-place
}

NdArray operator/=(NdArray& lhs, const NdArray& rhs) {
    return lhs = ApplyElemWiseOpInplace(std::move(lhs), rhs,
                                        std::divides<float>(),
                                        false);  // force in-place
}

// Compound Assignment (NdArray, float)
NdArray operator+=(NdArray& lhs, float rhs) {
    return lhs = Add(std::move(lhs), rhs);
}

NdArray operator-=(NdArray& lhs, float rhs) {
    return lhs = Subtract(std::move(lhs), rhs);
}

NdArray operator*=(NdArray& lhs, float rhs) {
    return lhs = Multiply(std::move(lhs), rhs);
}

NdArray operator/=(NdArray& lhs, float rhs) {
    return lhs = Divide(std::move(lhs), rhs);
}

// ---------------------------- Operator Functions -----------------------------
// Arithmetic operators (NdArray, NdArray)
NdArray Add(const NdArray& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::plus<float>());
}

NdArray Subtract(const NdArray& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::minus<float>());
}

NdArray Multiply(const NdArray& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::multiplies<float>());
}

NdArray Divide(const NdArray& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::divides<float>());
}

// Arithmetic operators (NdArray, float)
NdArray Add(const NdArray& lhs, float rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::plus<float>());
}

NdArray Subtract(const NdArray& lhs, float rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::minus<float>());
}

NdArray Multiply(const NdArray& lhs, float rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::multiplies<float>());
}

NdArray Divide(const NdArray& lhs, float rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::divides<float>());
}

// Arithmetic operators (float, NdArray)
NdArray Add(float lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::plus<float>());
}

NdArray Subtract(float lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::minus<float>());
}

NdArray Multiply(float lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::multiplies<float>());
}

NdArray Divide(float lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::divides<float>());
}

// Comparison operators (NdArray, NdArray)
NdArray Equal(const NdArray& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::equal_to<float>());
}

NdArray NotEqual(const NdArray& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::not_equal_to<float>());
}

NdArray Greater(const NdArray& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::greater<float>());
}

NdArray GreaterEqual(const NdArray& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::greater_equal<float>());
}

NdArray Less(const NdArray& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::less<float>());
}

NdArray LessEqual(const NdArray& lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::less_equal<float>());
}

// Comparison operators (NdArray, float)
NdArray Equal(const NdArray& lhs, float rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::equal_to<float>());
}

NdArray NotEqual(const NdArray& lhs, float rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::not_equal_to<float>());
}

NdArray Greater(const NdArray& lhs, float rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::greater<float>());
}

NdArray GreaterEqual(const NdArray& lhs, float rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::greater_equal<float>());
}

NdArray Less(const NdArray& lhs, float rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::less<float>());
}

NdArray LessEqual(const NdArray& lhs, float rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::less_equal<float>());
}

// Comparison operators (float, NdArray)
NdArray Equal(float lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::equal_to<float>());
}

NdArray NotEqual(float lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::not_equal_to<float>());
}

NdArray Greater(float lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::greater<float>());
}

NdArray GreaterEqual(float lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::greater_equal<float>());
}

NdArray Less(float lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::less<float>());
}

NdArray LessEqual(float lhs, const NdArray& rhs) {
    return ApplyElemWiseOp(lhs, rhs, std::less_equal<float>());
}

// Matrix operators
NdArray Dot(const NdArray& lhs, const NdArray& rhs) {
    return lhs.dot(rhs);
}

NdArray Dot(const NdArray& lhs, float rhs) {
    return lhs * rhs;  // Simple multiply
}

NdArray Dot(float lhs, const NdArray& rhs) {
    return lhs * rhs;  // Simple multiply
}

NdArray Cross(const NdArray& lhs, const NdArray& rhs) {
    return lhs.cross(rhs);
}

// Basic math operators
NdArray Abs(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::abs));
}

NdArray Ceil(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::ceil));
}

NdArray Floor(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::floor));
}

NdArray Sqrt(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::sqrt));
}

NdArray Exp(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::exp));
}

NdArray Log(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::log));
}

NdArray Power(const NdArray& x, const NdArray& y) {
    return ApplyElemWiseOp(x, y,
                           static_cast<float (*)(float, float)>(std::pow));
}

NdArray Power(const NdArray& x, float y) {
    return ApplyElemWiseOp(x, y,
                           static_cast<float (*)(float, float)>(std::pow));
}

NdArray Power(float x, const NdArray& y) {
    return ApplyElemWiseOp(x, y,
                           static_cast<float (*)(float, float)>(std::pow));
}

// Trigonometric functions
NdArray Sin(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::sin));
}

NdArray Cos(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::cos));
}

NdArray Tan(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::tan));
}

// Inverse trigonometric functions
NdArray ArcSin(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::asin));
}

NdArray ArcCos(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::acos));
}

NdArray ArcTan(const NdArray& x) {
    return ApplySingleOp(x, static_cast<float (*)(float)>(std::atan));
}

NdArray ArcTan2(const NdArray& y, const NdArray& x) {
    return ApplyElemWiseOp(y, x,
                           static_cast<float (*)(float, float)>(std::atan2));
}

NdArray ArcTan2(const NdArray& y, float x) {
    return ApplyElemWiseOp(y, x,
                           static_cast<float (*)(float, float)>(std::atan2));
}

NdArray ArcTan2(float y, const NdArray& x) {
    return ApplyElemWiseOp(y, x,
                           static_cast<float (*)(float, float)>(std::atan2));
}

// Axis functions
NdArray Sum(const NdArray& x, const Axes& axes) {
    return ReduceAxis(x, axes, 0.f, std::plus<float>());
}

NdArray Min(const NdArray& x, const Axes& axes) {
    return ReduceAxisNoEmpty(x, axes, std::numeric_limits<float>::max(),
                             [](float a, float b) { return std::min(a, b); });
}

NdArray Max(const NdArray& x, const Axes& axes) {
    return ReduceAxisNoEmpty(x, axes, -std::numeric_limits<float>::max(),
                             [](float a, float b) { return std::max(a, b); });
}

NdArray Mean(const NdArray& x, const Axes& axes) {
    if (x.size() == 0) {
        return {std::numeric_limits<float>::quiet_NaN()};
    }
    auto&& sum = Sum(x, axes);
    float denom = x.size() / sum.size();
    return sum / denom;
}

// Inverse
NdArray Inv(const NdArray& x) {
    return InvertNdArray(x);
}

// ------------------------ In-place Operator Functions ------------------------
// Arithmetic operators (NdArray, NdArray)
NdArray Add(NdArray&& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), std::move(rhs),
                                  std::plus<float>());
}

NdArray Add(const NdArray& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs), std::plus<float>());
}

NdArray Add(NdArray&& lhs, const NdArray& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs, std::plus<float>());
}

NdArray Subtract(NdArray&& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), std::move(rhs),
                                  std::minus<float>());
}

NdArray Subtract(const NdArray& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs), std::minus<float>());
}

NdArray Subtract(NdArray&& lhs, const NdArray& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs, std::minus<float>());
}

NdArray Multiply(NdArray&& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), std::move(rhs),
                                  std::multiplies<float>());
}

NdArray Multiply(const NdArray& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs),
                                  std::multiplies<float>());
}

NdArray Multiply(NdArray&& lhs, const NdArray& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs,
                                  std::multiplies<float>());
}

NdArray Divide(NdArray&& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), std::move(rhs),
                                  std::divides<float>());
}

NdArray Divide(const NdArray& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs), std::divides<float>());
}

NdArray Divide(NdArray&& lhs, const NdArray& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs, std::divides<float>());
}

// Arithmetic operators (NdArrarhs, float)
NdArray Add(NdArray&& lhs, float rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs, std::plus<float>());
}

NdArray Subtract(NdArray&& lhs, float rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs, std::minus<float>());
}

NdArray Multiply(NdArray&& lhs, float rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs,
                                  std::multiplies<float>());
}

NdArray Divide(NdArray&& lhs, float rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs, std::divides<float>());
}

// Arithmetic operators (float, NdArray)
NdArray Add(float lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs), std::plus<float>());
}

NdArray Subtract(float lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs), std::minus<float>());
}

NdArray Multiply(float lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs),
                                  std::multiplies<float>());
}

NdArray Divide(float lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs), std::divides<float>());
}

// Comparison operators (NdArrarhs, NdArray)
NdArray Equal(NdArray&& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), std::move(rhs),
                                  std::equal_to<float>());
}

NdArray Equal(const NdArray& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs), std::equal_to<float>());
}

NdArray Equal(NdArray&& lhs, const NdArray& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs, std::equal_to<float>());
}

NdArray NotEqual(NdArray&& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), std::move(rhs),
                                  std::not_equal_to<float>());
}

NdArray NotEqual(const NdArray& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs),
                                  std::not_equal_to<float>());
}

NdArray NotEqual(NdArray&& lhs, const NdArray& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs,
                                  std::not_equal_to<float>());
}

NdArray Greater(NdArray&& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), std::move(rhs),
                                  std::greater<float>());
}

NdArray Greater(const NdArray& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs), std::greater<float>());
}

NdArray Greater(NdArray&& lhs, const NdArray& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs, std::greater<float>());
}

NdArray GreaterEqual(NdArray&& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), std::move(rhs),
                                  std::greater_equal<float>());
}

NdArray GreaterEqual(const NdArray& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs),
                                  std::greater_equal<float>());
}

NdArray GreaterEqual(NdArray&& lhs, const NdArray& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs,
                                  std::greater_equal<float>());
}

NdArray Less(NdArray&& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), std::move(rhs),
                                  std::less<float>());
}

NdArray Less(const NdArray& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs), std::less<float>());
}

NdArray Less(NdArray&& lhs, const NdArray& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs, std::less<float>());
}

NdArray LessEqual(NdArray&& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), std::move(rhs),
                                  std::less_equal<float>());
}

NdArray LessEqual(const NdArray& lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs),
                                  std::less_equal<float>());
}

NdArray LessEqual(NdArray&& lhs, const NdArray& rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs,
                                  std::less_equal<float>());
}

// Comparison operators (NdArrarhs, float)
NdArray Equal(NdArray&& lhs, float rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs, std::equal_to<float>());
}

NdArray NotEqual(NdArray&& lhs, float rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs,
                                  std::not_equal_to<float>());
}

NdArray Greater(NdArray&& lhs, float rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs, std::greater<float>());
}

NdArray GreaterEqual(NdArray&& lhs, float rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs,
                                  std::greater_equal<float>());
}

NdArray Less(NdArray&& lhs, float rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs, std::less<float>());
}

NdArray LessEqual(NdArray&& lhs, float rhs) {
    return ApplyElemWiseOpInplace(std::move(lhs), rhs,
                                  std::less_equal<float>());
}

// Comparison operators (float, NdArray)
NdArray Equal(float lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs), std::equal_to<float>());
}

NdArray NotEqual(float lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs),
                                  std::not_equal_to<float>());
}

NdArray Greater(float lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs), std::greater<float>());
}

NdArray GreaterEqual(float lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs),
                                  std::greater_equal<float>());
}

NdArray Less(float lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs), std::less<float>());
}

NdArray LessEqual(float lhs, NdArray&& rhs) {
    return ApplyElemWiseOpInplace(lhs, std::move(rhs),
                                  std::less_equal<float>());
}

// Basic math operators
NdArray Abs(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::abs));
}

NdArray Ceil(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::ceil));
}

NdArray Floor(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::floor));
}

NdArray Sqrt(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::sqrt));
}

NdArray Exp(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::exp));
}

NdArray Log(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::log));
}

NdArray Power(NdArray&& x, NdArray&& y) {
    return ApplyElemWiseOpInplace(
            std::move(x), std::move(y),
            static_cast<float (*)(float, float)>(std::pow));
}

NdArray Power(const NdArray& x, NdArray&& y) {
    return ApplyElemWiseOpInplace(
            x, std::move(y), static_cast<float (*)(float, float)>(std::pow));
}

NdArray Power(NdArray&& x, const NdArray& y) {
    return ApplyElemWiseOpInplace(
            std::move(x), y, static_cast<float (*)(float, float)>(std::pow));
}

NdArray Power(NdArray&& x, float y) {
    return ApplyElemWiseOpInplace(
            std::move(x), y, static_cast<float (*)(float, float)>(std::pow));
}

NdArray Power(float x, NdArray&& y) {
    return ApplyElemWiseOpInplace(
            x, std::move(y), static_cast<float (*)(float, float)>(std::pow));
}

// Trigonometric functions
NdArray Sin(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::sin));
}

NdArray Cos(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::cos));
}

NdArray Tan(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::tan));
}

// Inverse trigonometric functions
NdArray ArcSin(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::asin));
}

NdArray ArcCos(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::acos));
}

NdArray ArcTan(NdArray&& x) {
    return ApplySingleOpInplace(std::move(x),
                                static_cast<float (*)(float)>(std::atan));
}

NdArray ArcTan2(NdArray&& y, NdArray&& x) {
    return ApplyElemWiseOpInplace(
            std::move(y), std::move(x),
            static_cast<float (*)(float, float)>(std::atan2));
}

NdArray ArcTan2(const NdArray& y, NdArray&& x) {
    return ApplyElemWiseOpInplace(
            y, std::move(x), static_cast<float (*)(float, float)>(std::atan2));
}

NdArray ArcTan2(NdArray&& y, const NdArray& x) {
    return ApplyElemWiseOpInplace(
            std::move(y), x, static_cast<float (*)(float, float)>(std::atan2));
}

NdArray ArcTan2(NdArray&& y, float x) {
    return ApplyElemWiseOpInplace(
            std::move(y), x, static_cast<float (*)(float, float)>(std::atan2));
}

NdArray ArcTan2(float y, NdArray&& x) {
    return ApplyElemWiseOpInplace(
            y, std::move(x), static_cast<float (*)(float, float)>(std::atan2));
}

// Inverse
NdArray Inv(NdArray&& x) {
    return InvertNdArrayInplace(std::move(x));
}

#endif  // TINYNDARRAY_IMPLEMENTATION

}  // namespace tinyndarray

#endif /* end of include guard */
