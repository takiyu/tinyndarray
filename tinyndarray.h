#ifndef TINYNDARRAY_H_ONCE
#define TINYNDARRAY_H_ONCE

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <thread>
#include <vector>

namespace tinyndarray {

class NdArray;
using InitShape = std::initializer_list<int>;
using Shape = std::vector<int>;
using Index = std::vector<int>;
using Axis = std::vector<int>;
using SliceIndex = std::vector<std::pair<int, int>>;
template <bool C>
using NdArrayT = typename std::conditional_t<C, const NdArray, NdArray>;
template <bool C>
using FloatT = typename std::conditional_t<C, const float, float>;

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
    template <bool C>
    class IterBase;
    using Iter = IterBase<false>;
    using ConstIter = IterBase<true>;

    NdArray();
    NdArray(const NdArray&);
    NdArray(NdArray&&) noexcept;
    NdArray& operator=(const NdArray&);
    NdArray& operator=(NdArray&&);
    virtual ~NdArray();

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

    static int GetNumWorkers();
    static void SetNumWorkers(int n_workers);  // -1: Hardware Concurrency
    static int GetBatchScale();
    static void SetBatchScale(int batch_scale);

    uintptr_t id() const;
    bool empty() const;
    size_t size() const;
    const Shape& shape() const;
    size_t ndim() const;
    void fill(float v);
    NdArray copy() const;

    Iter data();
    ConstIter data() const;
    Iter begin();
    Iter end();
    ConstIter begin() const;
    ConstIter end() const;

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

    bool isSlice() const;
    NdArray slice(const SliceIndex& slice_index) const;
    template <typename... I>
    NdArray slice(std::initializer_list<I>... slice_index) const;

    NdArray dot(const NdArray& other) const;
    NdArray dot(float other) const;
    NdArray cross(const NdArray& other) const;

    NdArray sum(const Axis& axes = {}) const;
    NdArray min(const Axis& axes = {}) const;
    NdArray max(const Axis& axes = {}) const;
    NdArray mean(const Axis& axes = {}) const;

    class Substance;

protected:
    std::shared_ptr<Substance> m_sub;
    NdArray(std::shared_ptr<Substance> sub);

    static std::random_device s_rand_seed;
    static std::mt19937 s_rand_engine;

    static int s_n_workers;
    static int s_batch_scale;
};

// --------------------------------- Iterator ----------------------------------
template <bool C>
class NdArray::IterBase {
public:
    IterBase(NdArrayT<C>& parent);
    IterBase(NdArrayT<C>& parent, FloatT<C>* ptr);
    ~IterBase();

    FloatT<C>& operator*();
    FloatT<C>& operator*() const;
    bool operator==(const IterBase& other) const;
    bool operator!=(const IterBase& other) const;
    IterBase operator+(int i) const;
    IterBase operator-(int i) const;
    FloatT<C>& operator[](int i);
    FloatT<C>& operator[](int i) const;
    IterBase& operator++();
    IterBase& operator--();
    IterBase& operator+=(int i);
    IterBase& operator-=(int i);
    operator IterBase<true>() const;  // Cast to const

private:
    void initPtrSlice();
    FloatT<C>* getForwardedPtr(int forward_cnt) const;
    FloatT<C>* getForwardedPtrSlice(int forward_cnt) const;

    std::function<FloatT<C>*(int)> m_get_forwarded_ptr;

    NdArrayT<C>& m_parent;
    FloatT<C>* m_ptr;
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
NdArray Sum(const NdArray& x, const Axis& axes = {});
NdArray Min(const NdArray& x, const Axis& axes = {});
NdArray Max(const NdArray& x, const Axis& axes = {});
NdArray Mean(const NdArray& x, const Axis& axes = {});
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

void GetPrallelParams(int size, int& n_workers, int& n_batch, int& batch_size) {
    // Fetch the number of workers
    n_workers = NdArray::GetNumWorkers();
    if (n_workers <= 0) {
        n_workers = static_cast<int>(std::thread::hardware_concurrency());
    }
    // Compute batch size and it number
    n_batch = n_workers * NdArray::GetBatchScale();
    batch_size = size / n_batch + (size % n_batch ? 1 : 0);
    n_workers = std::min(n_workers, batch_size);
}

template <typename F>
void RunParallel(int size, F op) {
    // Decide parallelization parameters
    int n_workers = -1, n_batch = -1, batch_size = -1;
    GetPrallelParams(size, n_workers, n_batch, batch_size);

    if (n_workers <= 1) {
        // Single execution
        for (int i = 0; i < size; i++) {
            // Operation
            op(i);
        }
    } else {
        // Parallel execution
        std::atomic<int> next_batch(0);
        std::vector<std::thread> workers(static_cast<size_t>(n_workers));
        for (auto&& worker : workers) {
            worker = std::thread([=, &next_batch]() {
                int batch_cnt = 0;
                while ((batch_cnt = next_batch++) < n_batch) {
                    for (int i = 0; i < batch_size; i++) {
                        const int idx = batch_size * batch_cnt + i;
                        if (size <= idx) {
                            break;
                        }
                        // Operation
                        op(idx);
                    }
                }
            });
        }
        for (auto&& worker : workers) {
            worker.join();
        }
    }
}

template <typename F, typename R>
float RunParallelWithReduce(int size, F op, R reduce, float init_v) {
    // Decide parallelization parameters
    int n_workers = -1, n_batch = -1, batch_size = -1;
    GetPrallelParams(size, n_workers, n_batch, batch_size);

    if (n_workers <= 1) {
        // Single execution
        float v = init_v;
        for (int i = 0; i < size; i++) {
            // Operation with reduction
            v = reduce(v, op(i));
        }
        return v;
    } else {
        // Parallel execution
        std::atomic<int> next_batch(0);
        std::vector<std::thread> workers(static_cast<size_t>(n_workers));
        std::vector<float> reduced_results(workers.size());
        for (size_t t = 0; t < workers.size(); t++) {
            workers[t] = std::thread([=, &next_batch, &reduced_results]() {
                int batch_cnt = 0;
                float v = init_v;
                while ((batch_cnt = next_batch++) < n_batch) {
                    for (int i = 0; i < batch_size; i++) {
                        const int idx = batch_size * batch_cnt + i;
                        if (size <= idx) {
                            break;
                        }
                        // Operation with reduction
                        v = reduce(v, op(idx));
                    }
                }
                reduced_results[t] = v;
            });
        }
        for (auto&& worker : workers) {
            worker.join();
        }
        return std::accumulate(reduced_results.begin(), reduced_results.end(),
                               init_v, reduce);
    }
}

template <typename Iter>
void FillN(Iter&& iter, const int n, float v) {
    // Fill in parallel
    RunParallel(n, [&](int i) { iter[i] = v; });
}

template <typename F>
inline void ApplyOpSimple(NdArray& ret, F op) {
    auto&& ret_data = ret.data();
    // Simply apply all
    RunParallel(static_cast<int>(ret.size()),
                [&](int i) { ret_data[i] = op(); });
}

template <typename F>
inline void ApplyOpSimple(NdArray& ret, const NdArray& src, F op) {
    auto&& ret_data = ret.data();
    auto&& src_data = src.data();
    // Simply apply all
    RunParallel(static_cast<int>(ret.size()),
                [&](int i) { ret_data[i] = op(src_data[i]); });
}

template <typename F>
inline void ApplyOpSimple(NdArray& ret, const NdArray& lhs, const NdArray& rhs,
                          F op) {
    auto&& ret_data = ret.data();
    auto&& l_data = lhs.data();
    auto&& r_data = rhs.data();
    // Simply apply all
    RunParallel(static_cast<int>(ret.size()),
                [&](int i) { ret_data[i] = op(l_data[i], r_data[i]); });
}

template <typename F>
inline void ApplyOpSimple(NdArray& ret, const NdArray& lhs, const float rhs,
                          F op) {
    auto&& ret_data = ret.data();
    auto&& l_data = lhs.data();
    // Simply apply all
    RunParallel(static_cast<int>(ret.size()),
                [&](int i) { ret_data[i] = op(l_data[i], rhs); });
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
    return std::move(x);
}

// ------------------ Utilities for NdArray (Broadcast common) -----------------
static Shape CheckBroadcastable(const Shape& l_shape, const Shape& r_shape) {
    // We assuming left array has deeper shape than right one.
    if (l_shape.size() < r_shape.size()) {
        return CheckBroadcastable(r_shape, l_shape);  // Swap
    }
    // `l_shape.size()` is maximum depth.

    // Check empty
    if (r_shape.size() == 1 && r_shape[0] == 0) {
        throw std::runtime_error("Broadcast of empty array");
    }

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

static size_t ReduceShapes(Shape& ret_shape, Shape& l_shape, Shape& r_shape,
                           const size_t depth_offset) {
    // Require `ret_shape.size() == l_shape.size() == r_shape.size()`

    // Remove meaningless dimensions.
    Shape ret_shape_cleaned, l_shape_cleaned, r_shape_cleaned;
    int size_pool = 1;
    size_t depth = 0;
    for (; depth < ret_shape.size() - depth_offset; depth++) {
        if (l_shape[depth] == r_shape[depth]) {
            // Store
            size_pool *= l_shape[depth];
        } else {
            // Pop
            if (size_pool != 1) {
                ret_shape_cleaned.push_back(size_pool);
                l_shape_cleaned.push_back(size_pool);
                r_shape_cleaned.push_back(size_pool);
                size_pool = 1;
            }
            // Through current dimension
            ret_shape_cleaned.push_back(ret_shape[depth]);
            l_shape_cleaned.push_back(l_shape[depth]);
            r_shape_cleaned.push_back(r_shape[depth]);
        }
    }
    // Pop
    if (size_pool != 1) {
        ret_shape_cleaned.push_back(size_pool);
        l_shape_cleaned.push_back(size_pool);
        r_shape_cleaned.push_back(size_pool);
    }
    // Store actual depth count
    const size_t n_depth = ret_shape_cleaned.size();
    // Pass through included in `depth_offset`.
    for (; depth < ret_shape.size(); depth++) {
        ret_shape_cleaned.push_back(ret_shape[depth]);
        l_shape_cleaned.push_back(l_shape[depth]);
        r_shape_cleaned.push_back(r_shape[depth]);
    }
    // Return
    ret_shape = std::move(ret_shape_cleaned);
    l_shape = std::move(l_shape_cleaned);
    r_shape = std::move(r_shape_cleaned);
    return n_depth;
}

template <std::size_t ret_step, typename F>
void ApplyOpBroadcastImpl(const NdArray::Iter& ret_data,
                          const NdArray::ConstIter& l_data,
                          const NdArray::ConstIter& r_data,
                          const Shape& ret_shape, const int ret_size,
                          const std::vector<int>& l_steps,
                          const std::vector<int>& r_steps,
                          const size_t start_depth, const size_t n_depth,
                          F op) {
    // Create stacks and counter
    std::vector<int> ret_cnts(n_depth);
    std::vector<int> l_idx_stack(n_depth), r_idx_stack(n_depth);
    size_t depth = start_depth;
    int l_idx = 0;
    int r_idx = 0;

    for (int ret_idx = 0; ret_idx < ret_size; ret_idx += ret_step) {
        // Go down
        for (; depth < n_depth; depth++) {
            l_idx_stack[depth] = l_idx;  // Push stack
            r_idx_stack[depth] = r_idx;
        }

        // Operate
        op(ret_data + ret_idx, l_data + l_idx, r_data + r_idx);

        // Go up and count
        for (; start_depth < depth; depth--) {
            const size_t prev_d = depth - 1;
            ret_cnts[prev_d]++;        // Count up
            l_idx += l_steps[prev_d];  // Forward index
            r_idx += r_steps[prev_d];
            if (ret_cnts[prev_d] < ret_shape[prev_d]) {
                break;  // Continue normally
            }
            // Go upper depth
            ret_cnts[prev_d] = 0;         // Clear count
            l_idx = l_idx_stack[prev_d];  // Pop stack
            r_idx = r_idx_stack[prev_d];
        }
    }
}

template <std::size_t ret_step, typename F>
void ApplyOpBroadcast(NdArray& ret, const NdArray& lhs, const NdArray& rhs,
                      const size_t depth_offset, F op) {
    Shape ret_shape = ret.shape();

    // Pre-compute padded shapes
    Shape l_shape = PadShape(lhs.shape(), ret_shape.size());
    Shape r_shape = PadShape(rhs.shape(), ret_shape.size());

    // Pre-compute reduced shapes
    const size_t n_depth =
            ReduceShapes(ret_shape, l_shape, r_shape, depth_offset);

    // Pre-compute child sizes
    const std::vector<int>& ret_child_sizes = ComputeChildSizes(ret_shape);
    const std::vector<int>& l_child_sizes = ComputeChildSizes(l_shape);
    const std::vector<int>& r_child_sizes = ComputeChildSizes(r_shape);

    // Pre-compute steps
    std::vector<int> l_steps, r_steps;
    l_steps.reserve(n_depth);
    r_steps.reserve(n_depth);
    for (size_t depth = 0; depth < n_depth; depth++) {
        const int& l_s = l_shape[depth];
        const int& r_s = r_shape[depth];
        const int l_step = (l_s == r_s || r_s == 1) ? l_child_sizes[depth] : 0;
        const int r_step = (l_s == r_s || l_s == 1) ? r_child_sizes[depth] : 0;
        l_steps.push_back(l_step);
        r_steps.push_back(r_step);
    }

#if 1  // Run in parallel
    RunParallel(ret_shape[0], [&](int i) {
        const int ret_size = static_cast<int>(ret.size()) / ret_shape[0];
        ApplyOpBroadcastImpl<ret_step>(
                ret.data() + ret_child_sizes[0] * i,
                lhs.data() + l_steps[0] * i, rhs.data() + r_steps[0] * i,
                ret_shape, ret_size, l_steps, r_steps, 1, n_depth, op);
    });
#else  // Run sequentially
    ApplyOpBroadcastImpl<ret_step>(ret.data(), lhs.data(), rhs.data(),
                                   ret_shape, static_cast<int>(ret.size()),
                                   l_steps, r_steps, 0, n_depth, op);
#endif
}

template <typename F>
inline auto WrapOpForIter(F op) {
    return [op](const NdArray::Iter& o, const NdArray::ConstIter& l,
                const NdArray::ConstIter& r) {
        *o = op(*l, *r);  // wrap pointer operation for iterator's one
    };
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
        ApplyOpBroadcast<1>(ret, lhs, rhs, 0, WrapOpForIter(op));
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
        return std::move(lhs);
    } else {
        // Check it is possible to broadcast
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        // Select result with shape matching
        NdArray&& ret =
                (ret_shape == lhs.shape()) ? lhs :                  // left
                        (ret_shape == rhs.shape()) ? rhs :          // right
                                (allow_new) ? NdArray(ret_shape) :  // new
                                        throw std::runtime_error(
                                                "Invalid shape for in-place"
                                                " operation");
        // Apply broadcast
        ApplyOpBroadcast<1>(ret, lhs, rhs, 0, WrapOpForIter(op));
        return std::move(ret);
    }
}

template <typename F>
NdArray ApplyElemWiseOpInplace(NdArray&& lhs, const NdArray& rhs, F op,
                               const bool allow_new = true) {
    if (lhs.shape() == rhs.shape()) {
        // Apply without broadcast because of same size for speed up.
        ApplyOpSimple(lhs, lhs, rhs, op);
        return std::move(lhs);
    } else {
        // Check it is possible to broadcast
        const Shape& ret_shape = CheckBroadcastable(lhs.shape(), rhs.shape());
        // Select result with shape matching
        NdArray&& ret =
                (ret_shape == lhs.shape()) ? lhs :          // left
                        (allow_new) ? NdArray(ret_shape) :  // new
                                throw std::runtime_error(
                                        "Invalid shape for in-place operation");
        // Apply broadcast (result matrix is lhs)
        ApplyOpBroadcast<1>(ret, lhs, rhs, 0, WrapOpForIter(op));
        return std::move(ret);
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
    return std::move(lhs);
}

template <typename F>
inline NdArray ApplyElemWiseOpInplace(float lhs, NdArray&& rhs, F op) {
    // Swap left and right
    return ApplyElemWiseOpInplace(std::move(rhs), lhs, ReverseOp(op));
}

// ------------------- Utilities for NdArray (Axis reduction) ------------------
static auto CheckReductable(const Shape& shape, const Axis& axes) {
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

static int ComputeReducedIndex(int src_idx,
                               const std::vector<int>& ret_child_sizes,
                               const std::vector<int>& src_child_sizes,
                               const Axis& sorted_axes) {
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

static void ReduceShapes(Shape& ret_shape, Shape& src_shape,
                         Axis& sorted_axes) {
    // Require `ret_shape.size() == src_shape.size()`

    // Remove meaningless dimensions.
    Shape ret_shape_cleaned, src_shape_cleaned;
    int size_pool = 1;
    size_t depth = 0;
    size_t axis_idx = 0;
    for (; depth < ret_shape.size(); depth++) {
        if (ret_shape[depth] == src_shape[depth]) {
            // Store
            size_pool *= ret_shape[depth];
        } else {
            // Pop
            if (size_pool != 1) {
                ret_shape_cleaned.push_back(size_pool);
                src_shape_cleaned.push_back(size_pool);
                size_pool = 1;
            }
            // Through current dimension
            ret_shape_cleaned.push_back(ret_shape[depth]);
            src_shape_cleaned.push_back(src_shape[depth]);
            // Adjust axis
            sorted_axes[axis_idx++] =
                    static_cast<int>(ret_shape_cleaned.size()) - 1;
        }
    }
    // Pop
    if (size_pool != 1) {
        ret_shape_cleaned.push_back(size_pool);
        src_shape_cleaned.push_back(size_pool);
    }
    if (axis_idx != sorted_axes.size() - 1) {
        sorted_axes[axis_idx] = static_cast<int>(ret_shape_cleaned.size()) - 1;
    }
    // Return
    ret_shape = std::move(ret_shape_cleaned);
    src_shape = std::move(src_shape_cleaned);
}

template <typename F>
float ReduceAxisAll(const NdArray::ConstIter& data, const size_t size,
                    const float init_v, F reduce_op) {
    auto&& op = [&](int i) { return data[i]; };
    const float ret = RunParallelWithReduce(static_cast<int>(size), op,
                                            reduce_op, init_v);
    return ret;
}

template <typename F>
void ReduceAxisImpl(const NdArray::Iter& ret_data,
                    const NdArray::ConstIter& src_data,
                    const std::vector<int>& ret_child_sizes,
                    const std::vector<int>& src_child_sizes,
                    const Shape& src_shape, const int src_size,
                    const Axis& sorted_axes, F reduce_op) {
    // TODO: Replace with more effective implementation.

    // Common reduce function with src_idx
    auto&& reduce = [&](int src_idx) {
        // Result index
        const int ret_idx = ComputeReducedIndex(src_idx, ret_child_sizes,
                                                src_child_sizes, sorted_axes);
        // Reduce one source element
        float& ret_v = ret_data[ret_idx];
        ret_v = reduce_op(ret_v, src_data[src_idx]);
    };

    // Try top-dim parallelization.
    // When axis 0 will be reduced, it cannot be in parallel.
    if (sorted_axes[0] == 0) {
        // Run sequentially
        for (int src_idx = 0; src_idx < src_size; src_idx++) {
            reduce(src_idx);
        }
    } else {
        // Run in parallel
        const int& n_parallel = src_shape[0];
        RunParallel(n_parallel, [&](int para_idx) {
            const int n_sub = src_child_sizes[0];
            const int base_idx = para_idx * n_sub;
            for (int sub_idx = 0; sub_idx < n_sub; sub_idx++) {
                const int src_idx = base_idx + sub_idx;
                reduce(src_idx);
            }
        });
    }
}

template <typename F>
NdArray ReduceAxis(const NdArray& src, const Axis& axes, const float init_v,
                   F reduce_op) {
    if (axes.size() == 0) {
        // No Axis -> Reduce all
        return {ReduceAxisAll(src.data(), src.size(), init_v, reduce_op)};
    } else {
        // Check it is possible to reduce.
        Shape src_shape = src.shape();
        const auto& ret_shapes = CheckReductable(src_shape, axes);
        const Shape& ret_shape = std::get<0>(ret_shapes);
        Shape ret_shape_pad = std::get<1>(ret_shapes);

        // Sort axes
        Axis sorted_axes = axes;
        std::sort(sorted_axes.begin(), sorted_axes.end());

        // Remove extra dimensions of shapes
        ReduceShapes(ret_shape_pad, src_shape, sorted_axes);

        // Pre-compute child sizes
        const auto& ret_child_sizes = ComputeChildSizes(ret_shape_pad);
        const auto& src_child_sizes = ComputeChildSizes(src_shape);

        // Result array with value initialization
        NdArray ret(ret_shape, init_v);

        // Reduce
        ReduceAxisImpl(ret.data(), src.data(), ret_child_sizes, src_child_sizes,
                       src_shape, static_cast<int>(src.size()), sorted_axes,
                       reduce_op);

        return ret;
    }
}

template <typename F>
NdArray ReduceAxisNoEmpty(const NdArray& src, const Axis& axes,
                          const float init_v, F reduce_op) {
    // Check empty
    if (src.size() == 0) {
        throw std::runtime_error("zero-size array to reduction operation");
    }
    // Call normally
    return ReduceAxis(src, axes, init_v, reduce_op);
}

// ----------------------- Utilities for NdArray (Print) -----------------------
static void OutputArrayLine(std::ostream& os, const NdArray::ConstIter& data,
                            const int size) {
    os << "[";  // Begin of a line
    for (int i = 0; i < size; i++) {
        os << data[i];  // Output an element
        if (i == size - 1) {
            os << "]";  // End of a line
        } else {
            os << ", ";  // Splitter of an element
        }
    }
}

static void OutputArrayMultiDim(std::ostream& os,
                                const NdArray::ConstIter& data,
                                const Shape& shape,
                                const std::vector<int>& child_sizes,
                                size_t depth) {
    for (int i = 0; i < shape[depth]; i++) {
        // Heading
        if (i == 0) {
            os << "[";  // begin of array
        } else {
            for (size_t d = 0; d < depth + 1; d++) {  // array indent
                os << " ";
            }
        }

        // Output internal array
        const int& child_size = child_sizes[depth];
        ;
        if (depth == shape.size() - 2) {
            OutputArrayLine(os, data + child_size * i, shape[depth + 1]);
        } else {
            OutputArrayMultiDim(os, data + child_size * i, shape, child_sizes,
                                depth + 1);
        }

        // Tailing
        if (i == shape[depth] - 1) {
            os << "]";  // End of array
        } else {
            os << "," << std::endl;  // Splitter of array
        }
    }
}

static void OutputNdArray(std::ostream& os, const NdArray& x) {
    const int size = static_cast<int>(x.size());
    const Shape& shape = x.shape();
    const std::vector<int>& child_sizes = ComputeChildSizes(shape);

    if (size == 0 || shape.size() == 0) {
        // Empty
        os << "[]";
    } else if (shape.size() == 1) {
        // 1-dim
        OutputArrayLine(os, x.data(), size);
    } else {
        // Multi-dim
        OutputArrayMultiDim(os, x.data(), shape, child_sizes, 0);
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
    const size_t size = lhs.size();
    if (size != rhs.size()) {
        throw std::runtime_error("Invalid size for inner product of 1D");
    }
    // Inner product of vectors
    auto&& l_data = lhs.data();
    auto&& r_data = rhs.data();
    auto&& op = [&](int i) { return l_data[i] * r_data[i]; };
    // Run in parallel
    const float sum = RunParallelWithReduce(static_cast<int>(size), op,
                                            std::plus<float>(), 0.f);
    return {sum};
}

void DotNdArray1d2dImplColMajor(const NdArray::Iter& ret_data,
                                const NdArray::ConstIter& l_data,
                                const NdArray::ConstIter& r_data,
                                const int n_col, const int n_contract) {
    // Zero initialization
    FillN(ret_data, n_col, 0.f);
    // Col-major dot product
    int r_idx = 0;
    for (int l_idx = 0; l_idx < n_contract; l_idx++) {
        for (int col_cnt = 0; col_cnt < n_col; col_cnt++) {
            ret_data[col_cnt] += l_data[l_idx] * r_data[r_idx];
            r_idx++;
        }
    }
}

void DotNdArray1d2dImplRowMajor(const NdArray::Iter& ret_data,
                                const NdArray::ConstIter& l_data,
                                const NdArray::ConstIter& r_data,
                                const int n_col, const int n_contract) {
    // Row-major dot product
    for (int col_cnt = 0; col_cnt < n_col; col_cnt++) {
        float sum = 0.f;
        int r_idx = col_cnt;
        for (int l_idx = 0; l_idx < n_contract; l_idx++) {
            sum += l_data[l_idx] * r_data[r_idx];
            r_idx += n_col;
        }
        ret_data[col_cnt] = sum;
    }
}

static auto SelectDot1d2dOp(const size_t l_size, const size_t r_size) {
    // Switch row-major and col-major
    if (l_size < r_size) {
        return DotNdArray1d2dImplColMajor;
    } else {
        return DotNdArray1d2dImplRowMajor;
    }
}

template <typename F1d2d>
void DotNdArray2dImpl(const NdArray::Iter& ret_data,
                      const NdArray::ConstIter& l_data,
                      const NdArray::ConstIter& r_data, const int n_row,
                      const int n_col, const int n_contract, F1d2d op_1d2d) {
#if 1  // Run in parallel
    RunParallel(n_row, [&](int row_cnt) {
        const int ret_idx = n_col * row_cnt;
        const int l_idx = n_contract * row_cnt;
        op_1d2d(ret_data + ret_idx, l_data + l_idx, r_data, n_col, n_contract);
    });
#else  // Run sequentially
    int ret_idx = 0;
    int l_idx = 0;
    for (int row_cnt = 0; row_cnt < n_row; row_cnt++) {
        op_1d2d(ret_data + ret_idx, l_data + l_idx, r_data, n_col, n_contract);
        l_idx += n_contract;
        ret_idx += n_col;
    }
#endif
}

static NdArray DotNdArray2d(const NdArray& lhs, const NdArray& rhs) {
    const Shape& l_shape = lhs.shape();  // 2 == size
    const Shape& r_shape = rhs.shape();  // 2 == size
    if (l_shape[1] != r_shape[0]) {
        throw std::runtime_error("Invalid size for inner product of 2D");
    }
    // Inner product of 2D matrix
    const int& n_row = l_shape[0];
    const int& n_col = r_shape[1];
    const int& n_contract = l_shape[1];  // == r_shape[0]
    NdArray ret({n_row, n_col});
    DotNdArray2dImpl(ret.data(), lhs.data(), rhs.data(), n_row, n_col,
                     n_contract, SelectDot1d2dOp(lhs.size(), lhs.size()));
    return ret;
}

template <typename F1d2d>
void DotNdArrayNdMdImpl(const NdArray::Iter& ret_data,
                        const NdArray::ConstIter& l_data,
                        const NdArray::ConstIter& r_data, const int n_l,
                        const int n_r, const int ret_step, const int l_step,
                        const int r_step, F1d2d op_1d2d) {
    const int& n_contract = l_step;
    const int& n_col = ret_step;
    const int& ret_idx_base = n_r;
#if 1  // Run in parallel
    if (n_l < n_r) {
        RunParallel(n_r, [&](int r_cnt) {  // Right-hand side loop
            const int ret_step_base = ret_idx_base * ret_step;
            const int r_idx = r_cnt * r_step;
            int l_idx = 0;
            int ret_idx = r_cnt * ret_step;
            for (int l_cnt = 0; l_cnt < n_l; l_cnt++) {
                op_1d2d(ret_data + ret_idx, l_data + l_idx, r_data + r_idx,
                        n_col, n_contract);
                l_idx += l_step;
                ret_idx += ret_step_base;
            }
        });
    } else {
        RunParallel(n_l, [&](int l_cnt) {  // Left-hand side loop
            const int l_idx = l_cnt * l_step;
            int r_idx = 0;
            int ret_idx = l_cnt * ret_idx_base * ret_step;
            for (int r_cnt = 0; r_cnt < n_r; r_cnt++) {
                op_1d2d(ret_data + ret_idx, l_data + l_idx, r_data + r_idx,
                        n_col, n_contract);
                r_idx += r_step;
                ret_idx += ret_step;
            }
        });
    }
#else  // Run sequentially
    int l_idx = 0;
    int ret_idx0 = 0;
    for (int l_cnt = 0; l_cnt < n_l; l_cnt++) {
        int r_idx = 0;
        int ret_idx = ret_idx0 * ret_step;
        for (int r_cnt = 0; r_cnt < n_r; r_cnt++) {
            op_1d2d(ret_data + ret_idx, l_data + l_idx, r_data + r_idx, n_col,
                    n_contract);
            r_idx += r_step;
            ret_idx += ret_step;
        }
        l_idx += l_step;
        ret_idx0 += ret_idx_base;
    }
#endif
}

static NdArray DotNdArrayNdMd(const NdArray& lhs, const NdArray& rhs) {
    const Shape& l_shape = lhs.shape();  // 1 <= l.size
    const Shape& r_shape = rhs.shape();  // 2 <= r.size

    // The last axis of left and the second-to-last axis of right must be same.
    const int n_contract = l_shape.end()[-1];
    if (n_contract != r_shape.end()[-2]) {
        throw std::runtime_error("Invalid shape for dot product");
    }

    // Result shape
    Shape ret_shape(l_shape.begin(), l_shape.end() - 1);
    ret_shape.insert(ret_shape.end(), r_shape.begin(), r_shape.end() - 2);
    ret_shape.push_back(r_shape.back());
    // Result array
    NdArray ret(ret_shape);

    // Compute 2D shapes and steps
    //   [2, 3, (4)] [5, 6, (4), 7] -> [2, 3, 5, 6, 7]
    const int ret_step = r_shape.end()[-1];    // [2, 3, 5, 6, <7>]
    const int l_step = n_contract;             // [2, 3, <4>]
    const int r_step = n_contract * ret_step;  // [5, 6, <4>, <7>]

    const int n_l = static_cast<int>(lhs.size()) / l_step;
    const int n_r = static_cast<int>(rhs.size()) / r_step;  // [<5>, <6>, 4, 7]

    // Dot product
    DotNdArrayNdMdImpl(ret.data(), lhs.data(), rhs.data(), n_l, n_r, ret_step,
                       l_step, r_step, SelectDot1d2dOp(lhs.size(), rhs.size()));

    return ret;
}

// ------------------- Utilities for NdArray (Cross product) -------------------
static void CrossNdArray1d1dShape33(const NdArray::Iter& ret_data,
                                    const NdArray::ConstIter& l_data,
                                    const NdArray::ConstIter& r_data) {
    // lhs.shape() == {3} && rhs.shape == {3}
    ret_data[0] = l_data[1] * r_data[2] - l_data[2] * r_data[1];
    ret_data[1] = l_data[2] * r_data[0] - l_data[0] * r_data[2];
    ret_data[2] = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

static void CrossNdArray1d1dShape32(const NdArray::Iter& ret_data,
                                    const NdArray::ConstIter& l_data,
                                    const NdArray::ConstIter& r_data) {
    // lhs.shape() == {3} && rhs.shape == {2}
    ret_data[0] = -l_data[2] * r_data[1];
    ret_data[1] = l_data[2] * r_data[0];
    ret_data[2] = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

static void CrossNdArray1d1dShape23(const NdArray::Iter& ret_data,
                                    const NdArray::ConstIter& l_data,
                                    const NdArray::ConstIter& r_data) {
    // lhs.shape() == {3} && rhs.shape == {3}
    ret_data[0] = l_data[1] * r_data[2];
    ret_data[1] = -l_data[0] * r_data[2];
    ret_data[2] = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

static void CrossNdArray1d1dShape22(const NdArray::Iter& ret_data,
                                    const NdArray::ConstIter& l_data,
                                    const NdArray::ConstIter& r_data) {
    // lhs.shape() == {2} && rhs.shape == {2}
    ret_data[0] = l_data[0] * r_data[1] - l_data[1] * r_data[0];
}

template <std::size_t ret_step, typename F>
NdArray CrossNdArrayNdMd(const NdArray& lhs, const NdArray& rhs, F op) {
    const Shape& l_shape = lhs.shape();
    const Shape& r_shape = rhs.shape();
    Shape ret_shape = CheckBroadcastable({l_shape.begin(), l_shape.end() - 1},
                                         {r_shape.begin(), r_shape.end() - 1});
    ret_shape.push_back(ret_step);
    // Apply broadcast
    NdArray ret(ret_shape);
    ApplyOpBroadcast<ret_step>(ret, lhs, rhs, 1, op);
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

static void InvertNdArray2d(NdArray::Iter ret_data, NdArray::ConstIter src_data,
                            int order) {
    const int order_2 = order * 2;
    const size_t tmp_size = static_cast<size_t>(order_2 * order_2);
    std::unique_ptr<float[]> tmp(new float[tmp_size]);
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

    int ret_idx = 0;
    for (int row = 0; row < order; row++) {
        for (int col = order; col < order_2; col++) {
            ret_data[ret_idx++] = tmp_data[row * order_2 + col];
        }
    }
}

static void InvertNdArrayNd(NdArray::Iter ret_data, NdArray::ConstIter src_data,
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
    return std::move(src);
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
    // Common variables
    size_t size = 0;
    Shape shape = {0};
    std::shared_ptr<float> v;  // C++17: Replace with `shared_ptr<float[]>`.
    // Slice variables
    bool is_slice = false;
    Index offset;
    Shape raw_shape;
};

// --------------------------------- Iterator ----------------------------------
template <bool C>
NdArray::IterBase<C>::IterBase(NdArrayT<C>& parent)
    : m_parent(parent), m_ptr(parent.m_sub->v.get()) {
//         if (parent.isSlice()) {
//             // Init as sliced NdArray
//             initPtrSlice();
//             m_get_forwarded_ptr = std::bind(&IterBase::getForwardedPtrSlice,
//                                             std::ref(*this),
//                                             std::placeholders::_1);
//         } else {
            // Init as normal NdArray
            m_get_forwarded_ptr = std::bind(&IterBase::getForwardedPtr,
                                            std::ref(*this),
                                            std::placeholders::_1);
//         }
}

template <bool C>
NdArray::IterBase<C>::IterBase(NdArrayT<C>& parent, FloatT<C>* ptr)
    : m_parent(parent), m_ptr(ptr) {
    // TODO
//     if (parent.isSlice()) {
//         // Init as sliced NdArray
// //         initPtrSlice();
//         m_get_forwarded_ptr = std::bind(&IterBase::getForwardedPtrSlice,
//                                         std::ref(*this),
//                                         std::placeholders::_1);
//     } else {
        // Init as normal NdArray
        m_get_forwarded_ptr = std::bind(&IterBase::getForwardedPtr,
                                        std::ref(*this),
                                        std::placeholders::_1);
//     }
}

template <bool C>
NdArray::IterBase<C>::~IterBase() = default;

template <bool C>
FloatT<C>& NdArray::IterBase<C>::operator*() {
    return *m_ptr;
}

template <bool C>
FloatT<C>& NdArray::IterBase<C>::operator*() const {
    return *m_ptr;
}

template <bool C>
bool NdArray::IterBase<C>::operator==(const IterBase<C>& other) const {
    return m_ptr == other.m_ptr;
}

template <bool C>
bool NdArray::IterBase<C>::operator!=(const IterBase<C>& other) const {
    return m_ptr != other.m_ptr;
}

template <bool C>
NdArray::IterBase<C> NdArray::IterBase<C>::operator+(int i) const {
    return {m_parent, m_ptr + i};
}

template <bool C>
NdArray::IterBase<C> NdArray::IterBase<C>::operator-(int i) const {
    return {m_parent, m_ptr - i};
}

template <bool C>
FloatT<C>& NdArray::IterBase<C>::operator[](int i) {
//     return *m_get_forwarded_ptr(i);
    return *getForwardedPtr(i);
}

template <bool C>
FloatT<C>& NdArray::IterBase<C>::operator[](int i) const {
//     return *m_get_forwarded_ptr(i);
    return *getForwardedPtr(i);
}

template <bool C>
NdArray::IterBase<C>& NdArray::IterBase<C>::operator++() {
//     m_ptr = m_get_forwarded_ptr(1);
    m_ptr = getForwardedPtr(1);
    return *this;
}

template <bool C>
NdArray::IterBase<C>& NdArray::IterBase<C>::operator--() {
//     m_ptr = m_get_forwarded_ptr(-1);
    m_ptr = getForwardedPtr(-1);
    return *this;
}

template <bool C>
NdArray::IterBase<C>& NdArray::IterBase<C>::operator+=(int i) {
//     m_ptr = m_get_forwarded_ptr(i);
    m_ptr = getForwardedPtr(i);
    return *this;
}

template <bool C>
NdArray::IterBase<C>& NdArray::IterBase<C>::operator-=(int i) {
//     m_ptr = m_get_forwarded_ptr(-i);
    m_ptr = getForwardedPtr(-i);
    return *this;
}

template <bool C>
NdArray::IterBase<C>::operator NdArray::IterBase<true>() const {
    return IterBase<true>{m_parent, m_ptr};
}

template <bool C>
void NdArray::IterBase<C>::initPtrSlice() {
    const auto& sub = *(m_parent.m_sub);

    // Compute offset's index
    int idx = 0;
    for (size_t i = 0; i < sub.offset.size(); i++) {
        idx *= sub.raw_shape[i];
        idx += sub.offset[i];
    }
    // Set pointer
    m_ptr = sub.v.get() + idx;
}

template <bool C>
FloatT<C>* NdArray::IterBase<C>::getForwardedPtr(int forward_cnt) const {
    return m_ptr + forward_cnt;
}

template <bool C>
FloatT<C>* NdArray::IterBase<C>::getForwardedPtrSlice(int forward_cnt) const {
    const auto& sub = *(m_parent.m_sub);

    auto&& shape = sub.shape;
    auto&& offset = sub.offset;
    auto&& raw_shape = sub.raw_shape;
    const size_t n_dim = raw_shape.size();
    const size_t last_dim = n_dim - 1;

    // Create decomposed index
    int cur_idx = static_cast<int>(m_ptr - sub.v.get());
    Index index(n_dim);
    for (size_t i = 0; i < n_dim - 1; i++) {
        index[i] = cur_idx / raw_shape[i + 1] - offset[i];
        cur_idx %= raw_shape[i];
    }
    index[last_dim] = cur_idx - offset[last_dim];

    // Add forward number
    int pseudo_idx = 0;
    for (size_t i = 0; i < n_dim; i++) {
        pseudo_idx *= shape[i];
        pseudo_idx += index[i];
    }
    pseudo_idx += forward_cnt;
    for (size_t i = 0; i < n_dim - 1; i++) {
        index[i] = pseudo_idx / shape[i + 1];
        pseudo_idx %= shape[i];
    }
    index[last_dim] = pseudo_idx;

    // Compose index
    int new_idx = 0;
    for (size_t i = 0; i < n_dim; i++) {
        new_idx *= raw_shape[i];
        new_idx += index[i] + offset[i];
    }

    // Add root pointer
    return sub.v.get() + new_idx;
}

// ------------------------- Sliced Iterator override --------------------------
// template <bool C>
// NdArray::SliceIterBase<C>::SliceIterBase(SliceNdArrayT<C>& parent)
//     : m_parent(parent), m_ptr(parent.m_sub->v.get()) {
//     initPtr();  // Set pointer
// }
//
// template <bool C>
// NdArray::SliceIterBase<C>::SliceIterBase(SliceNdArrayT<C>& parent, FloatT<C>* ptr)
//     : IterBase<C>(parent, ptr), m_slice_parent(parent) {}
//
// template <bool C>
// FloatT<C>& NdArray::SliceIterBase<C>::operator[](int i) {
//     return *getForwardedPtr(i);
// }
//
// template <bool C>
// FloatT<C>& NdArray::SliceIterBase<C>::operator[](int i) const {
//     return *getForwardedPtr(i);
// }
//
// template <bool C>
// NdArray::SliceIterBase<C>& NdArray::SliceIterBase<C>::operator++() {
//     this->m_ptr = getForwardedPtr(1);
//     return *this;
// }
//
// template <bool C>
// NdArray::SliceIterBase<C>& NdArray::SliceIterBase<C>::operator--() {
//     this->m_ptr = getForwardedPtr(-1);
//     return *this;
// }
//
// template <bool C>
// NdArray::SliceIterBase<C>& NdArray::SliceIterBase<C>::operator+=(int i) {
//     this->m_ptr = getForwardedPtr(i);
//     return *this;
// }
//
// template <bool C>
// NdArray::SliceIterBase<C>& NdArray::SliceIterBase<C>::operator-=(int i) {
//     this->m_ptr = getForwardedPtr(-i);
//     return *this;
// }

// ------------------- Template Specializations of Iterators -------------------
template class NdArray::IterBase<false>;
template class NdArray::IterBase<true>;

// ------------------------------- Static Member -------------------------------
std::random_device NdArray::s_rand_seed;
std::mt19937 NdArray::s_rand_engine(s_rand_seed());
int NdArray::s_n_workers = -1;
int NdArray::s_batch_scale = 4;

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
    // Create empty array
    const int n = static_cast<int>(std::ceil((stop - start) / step));
    NdArray ret({n});
    // Fill by step
    auto&& data = ret.data();
    RunParallel(n,
                [&](int i) { data[i] = start + step * static_cast<float>(i); });
    return ret;
}

// ------------------------- Static Methods for Random -------------------------
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

// ------------------------ Static Methods for Parallel ------------------------
int NdArray::GetNumWorkers() {
    return s_n_workers;
}

void NdArray::SetNumWorkers(int n_workers) {
    s_n_workers = n_workers;
}

int NdArray::GetBatchScale() {
    return s_batch_scale;
}

void NdArray::SetBatchScale(int batch_scale) {
    s_batch_scale = batch_scale;
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

void NdArray::fill(float v) {
    FillN(data(), static_cast<int>(size()), v);
}

NdArray NdArray::copy() const {
    // Create completely new substance
    NdArray ret(this->shape());
    // Copy array data
    auto&& ret_data = ret.data();
    auto&& src_data = data();
    const int n = static_cast<int>(size());
    RunParallel(n, [&](int i) { ret_data[i] = src_data[i]; });
    return ret;
}

// ----------------------------- Iterator Methods ------------------------------
NdArray::Iter NdArray::data() {
    return Iter(*this);
}

NdArray::ConstIter NdArray::data() const {
    return ConstIter(*this);
}

NdArray::Iter NdArray::begin() {
    return Iter(*this);
}

NdArray::Iter NdArray::end() {
    return Iter(*this, m_sub->v.get() + m_sub->size);
}

NdArray::ConstIter NdArray::begin() const {
    return ConstIter(*this);
}

NdArray::ConstIter NdArray::end() const {
    return ConstIter(*this, m_sub->v.get() + m_sub->size);
}

// // --------------------- Iterator Methods for SliceNdArray ---------------------
// NdArray::SliceIter SliceNdArray::data() {
//     return SliceIter(*this);
// }
//
// NdArray::SliceConstIter SliceNdArray::data() const {
//     return SliceConstIter(*this);
// }
//
// NdArray::SliceIter SliceNdArray::begin() {
//     return SliceIter(*this);
// }
//
// NdArray::SliceIter SliceNdArray::end() {
//     return SliceIter(*this, m_sub->v.get() + m_sub->size);  // TODO
// }
//
// NdArray::SliceConstIter SliceNdArray::begin() const {
//     return SliceConstIter(*this);
// }
//
// NdArray::SliceConstIter SliceNdArray::end() const {
//     return SliceConstIter(*this, m_sub->v.get() + m_sub->size);  // TODO
// }

// ------------------------------- Cast Operator -------------------------------
NdArray::operator float() const {
    if (size() != 1) {
        throw std::runtime_error("Only size-1 arrays can be casted to float");
    }
    return *(data());
}

// ------------------------------- Index Methods -------------------------------
float& NdArray::operator[](int i) {
    // Use the same implementation of constant method.
    return const_cast<float&>(static_cast<const NdArray&>(*this)[i]);
}

const float& NdArray::operator[](int i) const {
    // Make the index positive
    const int p_idx = (0 <= i) ? i : static_cast<int>(size()) + i;
    // Direct access with range check
    return data()[p_idx];
}

float& NdArray::operator[](const Index& index) {
    // Use the same implementation of constant method.
    return const_cast<float&>(static_cast<const NdArray&>(*this)[index]);
}

const float& NdArray::operator[](const Index& index) const {
    const auto& shape = this->shape();
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
    return *(data() + i);
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
    // If slice, reshape copied array
    if (m_sub->is_slice) {
        return copy().reshape(shape);
    }

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
        if (this->size() != size) {
            std::stringstream ss;
            ss << "Invalid reshape (" << this->size() << "->" << size << ")";
            throw std::runtime_error(ss.str());
        }
    } else {
        if (this->size() % size != 0) {
            throw std::runtime_error("Invalid reshape (-1)");
        }
        new_shape[unknown_idx] = static_cast<int>(this->size() / size);
    }

    // Create reshaped array
    auto ret_sub = std::make_shared<Substance>(*m_sub);  // Deep copy except `v`
    ret_sub->shape = std::move(new_shape);  // Overwrite by new shape
    NdArray ret(ret_sub);
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
bool NdArray::isSlice() const {
    return m_sub->is_slice;
}

NdArray NdArray::slice(const SliceIndex& slice_index) const {
    const Shape& shape = this->shape();

    // Compute slice shape and new positive index
    Shape slice_shape;
    Index slice_offset;
    for (size_t i = 0; i < shape.size(); i++) {
        const auto& si = slice_index[i];
        if (slice_index.size() <= i) {
            // All
            slice_shape.push_back(shape[i]);
            slice_offset.push_back(0);
        } else {
            // Make index positive
            int s = (0 <= si.first) ? si.first : shape[i] + si.first;
            int e = (0 <= si.second) ? si.second : shape[i] + si.second;
            // Clamp
            s = Clamp(s, 0, shape[i] - 1);  // Start must be in range.
            e = Clamp(e, 0, shape[i]);      // End can be next of the last.
            // Register
            slice_shape.push_back(e - s);
            slice_offset.push_back(s);
        }
    }
    // Compute slice size
    const size_t slice_size = static_cast<size_t>(std::accumulate(
            slice_shape.begin(), slice_shape.end(), 1, std::multiplies<int>()));

    // Create sliced array
    auto ret_sub = std::make_shared<Substance>();
    // Set slice variables
    if (m_sub->is_slice) {                                  // Sub slice
        for (size_t i = 0; i < slice_offset.size(); i++) {  // Shift offset
            ret_sub->offset[i] += slice_offset[i];
        }
    } else {  // New slice
        ret_sub->is_slice = true;
        ret_sub->offset = slice_offset;
        ret_sub->raw_shape = m_sub->shape;
    }
    // Set common variables
    ret_sub->size = slice_size;
    ret_sub->shape = slice_shape;
    ret_sub->v = m_sub->v;

    // Return new array
    NdArray ret(ret_sub);
    return ret;
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
            return CrossNdArrayNdMd<3>(lhs, rhs, CrossNdArray1d1dShape33);
        } else if (l_back == 3 && r_back == 2) {  // [2].cross([3]) -> [3]
            return CrossNdArrayNdMd<3>(lhs, rhs, CrossNdArray1d1dShape32);
        } else if (l_back == 2 && r_back == 3) {  // [3].cross([2]) -> [3]
            return CrossNdArrayNdMd<3>(lhs, rhs, CrossNdArray1d1dShape23);
        } else if (l_back == 2 && r_back == 2) {  // [2].cross([2]) -> [1]
            auto&& ret = CrossNdArrayNdMd<1>(lhs, rhs, CrossNdArray1d1dShape22);
            const Shape& ret_shape = ret.shape();  // Remove last dim
            return ret.reshape(Shape{ret_shape.begin(), ret_shape.end() - 1});
        }
    }
    throw std::runtime_error(
            "incompatible dimensions for cross product"
            " (dimension must be 2 or 3)");
}

// -------------------------------- Axis Method --------------------------------
NdArray NdArray::sum(const Axis& axes) const {
    return Sum(*this, axes);
}

NdArray NdArray::min(const Axis& axes) const {
    return Min(*this, axes);
}

NdArray NdArray::max(const Axis& axes) const {
    return Max(*this, axes);
}

NdArray NdArray::mean(const Axis& axes) const {
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
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII,
                                ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII, ISII,
                                ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII, ISII,
                                ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII, ISII,
                                ISII, ISII, ISII) const;
template NdArray NdArray::slice(ISII, ISII, ISII, ISII, ISII, ISII, ISII,
                                     ISII, ISII, ISII, ISII) const;

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
NdArray Sum(const NdArray& x, const Axis& axes) {
    return ReduceAxis(x, axes, 0.f, std::plus<float>());
}

NdArray Min(const NdArray& x, const Axis& axes) {
    return ReduceAxisNoEmpty(x, axes, std::numeric_limits<float>::max(),
                             [](float a, float b) { return std::min(a, b); });
}

NdArray Max(const NdArray& x, const Axis& axes) {
    return ReduceAxisNoEmpty(x, axes, -std::numeric_limits<float>::max(),
                             [](float a, float b) { return std::max(a, b); });
}

NdArray Mean(const NdArray& x, const Axis& axes) {
    if (x.size() == 0) {
        return {std::numeric_limits<float>::quiet_NaN()};
    }
    auto&& sum = Sum(x, axes);
    return sum / static_cast<float>(x.size() / sum.size());
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
