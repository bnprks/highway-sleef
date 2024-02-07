// Copyright 2020 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdint.h>
#include <stdio.h>

#include <cfloat>  // FLT_MAX
#include <cmath>   // std::abs
#include <random>
#include <utility>  // std::pair

// clang-format off
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/tests/sampling_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/tests/test_util-inl.h"
#include "src/gen-bindings/sleef-generated.h"
// clang-format on

#ifndef WIDER_FLOAT_IMPL
#define WIDER_FLOAT_IMPL
namespace hwy {
namespace detail {

// Note: We may need to add some special logic for MSVC at some point,
// since its long double type is basically just a double.
template <typename T>
struct WiderFloatImpl;
template <>
struct WiderFloatImpl<float> {
  using Type = double;
};
template <>
struct WiderFloatImpl<double> {
  using Type = long double;
};

template <typename T>
using WiderFloat = typename WiderFloatImpl<T>::Type;

template <typename T>
double ulp_diff(WiderFloat<T> expected, T actual) {
  if (std::isinf(expected) && static_cast<T>(expected) == actual) return 0;
  if (std::isnan(expected) || std::isnan(actual))
    return 0;  // Handle NaN mismatches elsewhere
  if (static_cast<T>(expected) == 0 && actual == 0) return 0;

  WiderFloat<T> diff = std::abs(expected - static_cast<WiderFloat<T>>(actual));

  WiderFloat<T> ulp = std::numeric_limits<T>::denorm_min();
  if (expected != 0) {
    // Calculate lowerp-precision expected1 and expected2 that straddle
    // the actual value of expected
    T expected1 = expected;
    T expected2 =
        std::nextafter(expected1, (expected1 < expected ? 1 : 1) *
                                      std::numeric_limits<T>::infinity());
    ulp = std::abs(static_cast<WiderFloat<T>>(expected1) -
                   static_cast<WiderFloat<T>>(expected2));
  }

  return static_cast<double>(diff / ulp);
}

}  // namespace detail
}  // namespace hwy
#endif

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

template <class T, class D>
HWY_NOINLINE void TestMath(
    const char* name,
    ::hwy::detail::WiderFloat<T> (*fx1)(::hwy::detail::WiderFloat<T>),
    Vec<D> (*fxN)(D, VecArg<Vec<D>>), D d, T min, T max, double max_error_ulp,
    std::initializer_list<T> special_values = {}) {
  // Overall strategy:
  // - Choose 32K uniform random unsigned ints, and convert them to float/double
  // inputs
  //   - Even for double-precision inputs, this should cover each exponent value
  //     with very high probability
  // - For all numbers in the "special inputs", also run a range of 64 numbers
  // spanning each side
  //   - Special inputs will include NaN, -Inf, Inf, -1, -0, 0, 1, FLT_MIN,
  //   FLT_MAX, min, max, and
  //     any additional user-specified values.
  // Passing criteria:
  // - Inputs in range [min, max] should have error <= max_error_ulp
  // - All inputs should result in NaN if and only if the reference function
  // returns NaN

  // Aim for roughly 1ms of test time given slow ops that take ~35ns / float
  constexpr size_t kSamples = AdjustedReps(1 << 15);
  constexpr size_t kValuesPerSpecial = 64;

  std::vector<T> kSpecialValues = {
      std::numeric_limits<T>::quiet_NaN(),
      -std::numeric_limits<T>::infinity(),
      std::numeric_limits<T>::infinity(),
      -1.0,
      1.0,
      -0.0,
      0.0,
      std::numeric_limits<T>::max(),
      std::numeric_limits<T>::lowest(),
  };
  kSpecialValues.insert(kSpecialValues.end(), special_values);

  size_t total_inputs =
      kSamples + (2 * kValuesPerSpecial + 1) * (kSpecialValues.size());

  // These should be under ~512 KB total, so don't worry about size
  std::unique_ptr<T[]> inputs = std::make_unique<T[]>(total_inputs);
  std::unique_ptr<T[]> outputs = std::make_unique<T[]>(total_inputs);

  // Fill in special input values
  size_t i = 0;
  for (const auto& x : kSpecialValues) {
    inputs[i++] = x;
    // Add upper neighbors
    T val = x;
    for (size_t j = 0; j < kValuesPerSpecial; j++) {
      val = std::nextafter(val, std::numeric_limits<T>::infinity());
      inputs[i++] = val;
    }
    // Add lower neighbors
    val = x;
    for (size_t j = 0; j < kValuesPerSpecial; j++) {
      val = std::nextafter(val, -std::numeric_limits<T>::infinity());
      inputs[i++] = val;
    }
  }

  // Fill in randomly-sampled input values
  {
    std::random_device r;
    std::mt19937_64 e(r());
    std::uniform_int_distribution<MakeUnsigned<T>> d;
    for (; i < total_inputs; i++) {
      inputs[i++] = BitCastScalar<T>(d(e));
    }
  }

  // Run our vector function on all the inputs
  for (i = 0; i + Lanes(d) < total_inputs; i += Lanes(d)) {
    Vec<D> v = LoadU(d, inputs.get() + i);
    v = fxN(d, v);
    StoreU(v, d, outputs.get() + i);
  }
  // Finish the tail in case our total number is not a multiple of Lanes(d)
  {
    Vec<D> v = LoadU(d, inputs.get() + total_inputs - Lanes(d));
    v = fxN(d, v);
    StoreU(v, d, outputs.get() + total_inputs - Lanes(d));
  }

  double max_ulp = -1.0;
  T worst_input, worst_output;
  bool preserve_nan = true;
  T nan_input, nan_output;

  // Check that all values are within accuracy
  for (size_t i = 0; i < total_inputs; i++) {
    ::hwy::detail::WiderFloat<T> expected =
        fx1(static_cast<::hwy::detail::WiderFloat<T>>(inputs[i]));
    T actual = outputs[i];
    if (inputs[i] <= max && inputs[i] >= min) {
      double ulp = ::hwy::detail::ulp_diff(expected, actual);
      if (ulp > max_ulp) {
        max_ulp = ulp;
        worst_input = inputs[i];
        worst_output = outputs[i];
      }
    }
    if (std::isnan(expected) != std::isnan(actual)) {
      preserve_nan = false;
      nan_input = inputs[i];
      nan_output = outputs[i];
    }
  }

  fprintf(stderr, "%s: %s max_ulp %g\n", hwy::TypeName(T(), Lanes(d)).c_str(),
          name, max_ulp);
  const char* ulp_error_fmt =
      "\tworst_input=%f (0x%x), worst_output=%f (0x%x)\n";
  const char* nan_error_fmt =
      "\tworst_input=%f (0x%x), worst_output=%f (0x%x)\n";
  if (std::is_same<T, double>::value) {
    ulp_error_fmt = "\tworst_input=%f (0x%lx), worst_output=%f (0x%lx)\n";
    nan_error_fmt = "\tworst_input=%f (0x%lx), worst_output=%f (0x%lx)\n";
  }
  if (max_ulp > max_error_ulp) {
    fprintf(stderr, ulp_error_fmt, worst_input,
            hwy::BitCastScalar<MakeUnsigned<T>>(worst_input), worst_output,
            hwy::BitCastScalar<MakeUnsigned<T>>(worst_output));
  }
  if (!preserve_nan) {
    fprintf(stderr, nan_error_fmt, nan_input,
            hwy::BitCastScalar<MakeUnsigned<T>>(nan_input), nan_output,
            hwy::BitCastScalar<MakeUnsigned<T>>(nan_output));
  }
  HWY_ASSERT(max_ulp <= max_error_ulp);
  HWY_ASSERT(preserve_nan);
}

// Overload for two-argument math functions
template <class T, class D>
HWY_NOINLINE void TestMath(
    const char* name,
    ::hwy::detail::WiderFloat<T> (*fx1)(::hwy::detail::WiderFloat<T>,
                                        ::hwy::detail::WiderFloat<T>),
    Vec<D> (*fxN)(D, VecArg<Vec<D>>, VecArg<Vec<D>>), D d, T min, T max,
    double max_error_ulp, std::initializer_list<T> special_values = {}) {
  // Overall strategy:
  // - Choose 32K uniform random pairs of unsigned ints, and convert them to
  // float/double
  //   inputs
  // - For all numbers in the "special inputs", also run a range of 4 numbers
  // spanning each side, and do all combinations of special inputs
  //   - Special inputs will include NaN, -Inf, Inf, -1, -0, 0, 1, FLT_MIN,
  //   FLT_MAX, min, max, and any additional user-specified values.
  //   - By default, this will be about 6.5K special value pairs
  // Passing criteria:
  // - Inputs in range [min, max] should have error <= max_error_ulp
  // - All inputs should result in NaN if and only if the reference function
  // returns NaN

  // Aim for roughly 1ms of test time given slow ops that take ~35ns / float
  constexpr size_t kSamples = AdjustedReps(1 << 15);
  constexpr size_t kValuesPerSpecial = 8;

  std::vector<T> kSpecialValues = {
      std::numeric_limits<T>::quiet_NaN(),
      -std::numeric_limits<T>::infinity(),
      std::numeric_limits<T>::infinity(),
      -1.0,
      1.0,
      -0.0,
      0.0,
      std::numeric_limits<T>::max(),
      std::numeric_limits<T>::lowest(),
  };
  kSpecialValues.insert(kSpecialValues.end(), special_values);

  // Fill in neighbors of special input values
  std::vector<T> all_special_values;
  for (const auto& x : kSpecialValues) {
    all_special_values.push_back(x);
    // Add upper neighbors
    T val = x;
    for (size_t j = 0; j < kValuesPerSpecial; j++) {
      val = std::nextafter(val, std::numeric_limits<T>::infinity());
      all_special_values.push_back(val);
    }
    // Add lower neighbors
    val = x;
    for (size_t j = 0; j < kValuesPerSpecial; j++) {
      val = std::nextafter(val, -std::numeric_limits<T>::infinity());
      all_special_values.push_back(val);
    }
  }

  size_t total_inputs =
      kSamples + all_special_values.size() * all_special_values.size();

  // These should be under 1 MB total, so don't worry about size
  std::unique_ptr<T[]> input1 = std::make_unique<T[]>(total_inputs);
  std::unique_ptr<T[]> input2 = std::make_unique<T[]>(total_inputs);
  std::unique_ptr<T[]> outputs = std::make_unique<T[]>(total_inputs);

  size_t i = 0;
  for (const auto& x : all_special_values) {
    for (const auto& y : all_special_values) {
      input1[i++] = x;
      input2[i++] = y;
    }
  }

  // Fill in randomly-sampled input values
  {
    std::random_device r;
    std::mt19937_64 e(r());
    std::uniform_int_distribution<MakeUnsigned<T>> d;
    for (; i < total_inputs; i++) {
      input1[i++] = BitCastScalar<T>(d(e));
      input2[i++] = BitCastScalar<T>(d(e));
    }
  }

  // Run our vector function on all the inputs
  for (i = 0; i + Lanes(d) < total_inputs; i += Lanes(d)) {
    Vec<D> v1 = LoadU(d, input1.get() + i);
    Vec<D> v2 = LoadU(d, input2.get() + i);
    Vec<D> res = fxN(d, v1, v2);
    StoreU(res, d, outputs.get() + i);
  }
  // Finish the tail in case our total number is not a multiple of Lanes(d)
  {
    Vec<D> v1 = LoadU(d, input1.get() + total_inputs - Lanes(d));
    Vec<D> v2 = LoadU(d, input2.get() + total_inputs - Lanes(d));
    Vec<D> res = fxN(d, v1, v2);
    StoreU(res, d, outputs.get() + total_inputs - Lanes(d));
  }

  double max_ulp = -1.0;
  std::pair<T, T> worst_input, nan_input;
  T worst_output, nan_output;
  bool preserve_nan = true;

  // Check that all values are within accuracy
  for (size_t i = 0; i < total_inputs; i++) {
    ::hwy::detail::WiderFloat<T> expected =
        fx1(static_cast<::hwy::detail::WiderFloat<T>>(input1[i]),
            static_cast<::hwy::detail::WiderFloat<T>>(input2[i]));
    T actual = outputs[i];
    if (input1[i] <= max && input1[i] >= min && input2[i] <= max &&
        input2[i] >= min) {
      double ulp = ::hwy::detail::ulp_diff(expected, actual);
      if (ulp > max_ulp) {
        max_ulp = ulp;
        worst_input = {input1[i], input2[i]};
        worst_output = outputs[i];
      }
    }
    if (std::isnan(expected) != std::isnan(actual)) {
      preserve_nan = false;
      nan_input = {input1[i], input2[i]};
      nan_output = outputs[i];
    }
  }

  fprintf(stderr, "%s: %s max_ulp %g\n", hwy::TypeName(T(), Lanes(d)).c_str(),
          name, max_ulp);
  const char* ulp_error_fmt =
      "\tworst_input=(%f (0x%x); %f (0x%x)), worst_output=%f (0x%x)\n";
  const char* nan_error_fmt =
      "\tworst_input=(%f (0x%x); %f (0x%x)), worst_output=%f (0x%x)\n";
  if (std::is_same<T, double>::value) {
    ulp_error_fmt = "\tworst_input=%f (0x%lx), worst_output=%f (0x%lx)\n";
    nan_error_fmt = "\tworst_input=%f (0x%lx), worst_output=%f (0x%lx)\n";
  }
  if (max_ulp > max_error_ulp) {
    fprintf(stderr, ulp_error_fmt, worst_input.first,
            hwy::BitCastScalar<MakeUnsigned<T>>(worst_input.first),
            worst_input.second,
            hwy::BitCastScalar<MakeUnsigned<T>>(worst_input.second),
            worst_output, hwy::BitCastScalar<MakeUnsigned<T>>(worst_output));
  }
  if (!preserve_nan) {
    fprintf(stderr, nan_error_fmt, nan_input.first,
            hwy::BitCastScalar<MakeUnsigned<T>>(nan_input.first),
            nan_input.second,
            hwy::BitCastScalar<MakeUnsigned<T>>(nan_input.second), nan_output,
            hwy::BitCastScalar<MakeUnsigned<T>>(nan_output));
  }
  HWY_ASSERT(max_ulp <= max_error_ulp);
  HWY_ASSERT(preserve_nan);
}

#define DEFINE_MATH_TEST_FUNC(NAME)                      \
  HWY_NOINLINE void TestAll##NAME() {                    \
    /*Test##NAME{}(double(), ScalableTag<double>{});  */ \
    ForFloat3264Types(ForPartialVectors<Test##NAME>());  \
  }

#undef DEFINE_MATH_TEST
#define DEFINE_MATH_TEST(NAME, F32x1, F32xN, F32_MIN, F32_MAX, F32_ERROR, \
                         F64x1, F64xN, F64_MIN, F64_MAX, F64_ERROR,       \
                         SPECIAL_VALS)                                    \
  struct Test##NAME {                                                     \
    template <class T, class D>                                           \
    HWY_NOINLINE void operator()(T, D d) {                                \
      if (sizeof(T) == 4) {                                               \
        TestMath<T, D>(HWY_STR(NAME), F32x1, F32xN, d, F32_MIN, F32_MAX,  \
                       F32_ERROR, SPECIAL_VALS);                          \
      } else {                                                            \
        TestMath<T, D>(HWY_STR(NAME), F64x1, F64xN, d,                    \
                       static_cast<T>(F64_MIN), static_cast<T>(F64_MAX),  \
                       F64_ERROR, SPECIAL_VALS);                          \
      }                                                                   \
    }                                                                     \
  };                                                                      \
  DEFINE_MATH_TEST_FUNC(NAME)

template <typename T>
T exp10_helper(T x);

template <>
float exp10_helper(float x) {
  return exp10f(x);
}

template <>
double exp10_helper(double x) {
  return exp10(x);
}

template <>
long double exp10_helper(long double x) {
  return exp10l(x);
}

// clang-format off
DEFINE_MATH_TEST(Exp,
  std::exp,   sleef::Exp,     -FLT_MAX,   FLT_MAX,    1,
  std::exp,   sleef::Exp,     -DBL_MAX,   DBL_MAX,    1,
  {})
DEFINE_MATH_TEST(Exp2,
  std::exp2,   sleef::Exp2,     -FLT_MAX,   FLT_MAX,    1,
  std::exp2,   sleef::Exp2,     -DBL_MAX,   DBL_MAX,    1,
  {})
DEFINE_MATH_TEST(Exp10,
  exp10_helper,    sleef::Exp10,     -FLT_MAX,   FLT_MAX,    1,
  exp10_helper,   sleef::Exp10,     -DBL_MAX,   DBL_MAX,    1,
  {})
DEFINE_MATH_TEST(Expm1,
  std::expm1, sleef::Expm1,   -FLT_MAX,   FLT_MAX,    1,
  std::expm1, sleef::Expm1,   -DBL_MAX,   DBL_MAX,    1,
  {})

DEFINE_MATH_TEST(Log,
  std::log,   sleef::Log,     -FLT_MAX,   FLT_MAX,    1,
  std::log,   sleef::Log,     -DBL_MAX,   DBL_MAX,    1,
  {})
DEFINE_MATH_TEST(LogFast,
  std::log,   sleef::LogFast, -FLT_MAX,   FLT_MAX,  3.5,
  std::log,   sleef::LogFast, -DBL_MAX,   DBL_MAX,  3.5,
  {})
DEFINE_MATH_TEST(Log2,
  std::log2,  sleef::Log2,    -FLT_MAX,   FLT_MAX,    1,
  std::log2,  sleef::Log2,    -DBL_MAX,   DBL_MAX,    1,
  {})
DEFINE_MATH_TEST(Log10,
  std::log10,  sleef::Log10,    -FLT_MAX,   FLT_MAX,    1,
  std::log10,  sleef::Log10,    -DBL_MAX,   DBL_MAX,    1,
  {})
DEFINE_MATH_TEST(Log1p,
  std::log1p, sleef::Log1p,   -FLT_MAX,     1e+38,    1,
  std::log1p, sleef::Log1p,   -DBL_MAX,    1e+307,    1,
  {})

DEFINE_MATH_TEST(Pow,
  std::pow,  sleef::Pow,    -FLT_MAX,   FLT_MAX,    1,
  std::pow,  sleef::Pow,    -DBL_MAX,   DBL_MAX,    1,
  {})

DEFINE_MATH_TEST(Sqrt,
  std::sqrt, sleef::Sqrt,   -FLT_MAX,   FLT_MAX,    1,
  std::sqrt, sleef::Sqrt,   -DBL_MAX,   DBL_MAX,    1,
  {})
DEFINE_MATH_TEST(SqrtFast,
  std::sqrt, sleef::SqrtFast,   -FLT_MAX,   FLT_MAX,    3.5,
  std::sqrt, sleef::SqrtFast,   -DBL_MAX,   DBL_MAX,    3.5,
  {})
DEFINE_MATH_TEST(Cbrt,
  std::cbrt, sleef::Cbrt,   -FLT_MAX,   FLT_MAX,    1,
  std::cbrt, sleef::Cbrt,   -DBL_MAX,   DBL_MAX,    1,
  {})
DEFINE_MATH_TEST(CbrtFast,
  std::cbrt, sleef::CbrtFast,   -FLT_MAX,   FLT_MAX,    3.5,
  std::cbrt, sleef::CbrtFast,   -DBL_MAX,   DBL_MAX,    3.5,
  {})
DEFINE_MATH_TEST(Hypot,
  std::hypot, sleef::Hypot,   -FLT_MAX,   FLT_MAX,    1,
  std::hypot, sleef::Hypot,   -DBL_MAX,   DBL_MAX,    1,
  {})
DEFINE_MATH_TEST(HypotFast,
  std::hypot, sleef::HypotFast,   -FLT_MAX,   FLT_MAX,    3.5,
  std::hypot, sleef::HypotFast,   -DBL_MAX,   DBL_MAX,    3.5,
  {})
// clang-format on

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {
HWY_BEFORE_TEST(HwySleefMathTest);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllExp);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllExp2);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllExp10);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllExpm1);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllLog);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllLogFast);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllLog2);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllLog10);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllLog1p);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllPow);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllSqrt);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllSqrtFast);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllCbrt);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllCbrtFast);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllHypot);
HWY_EXPORT_AND_TEST_P(HwySleefMathTest, TestAllHypotFast);
}  // namespace hwy

#endif
