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

#include <benchmark/benchmark.h>
#include <stdio.h>

#include <algorithm>
#include <cfloat>  // FLT_MAX
#include <cmath>   // std::abs
#include <limits>
#include <memory>
#include <random>

#include "define_dispatchers.h"

// clang-format off
//#undef HWY_TARGET_INCLUDE
//#define HWY_TARGET_INCLUDE "sleef_test.cc"
//#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/tests/test_util.h"
// clang-format on

static void fill_float(float *x, size_t n, float min, float max) {
  std::mt19937 gen(125124);  // Standard mersenne_twister_engine with seed
  std::uniform_real_distribution<> dis(min, max);
  for (size_t i = 0; i < n; i++) x[i] = dis(gen);
}

template <class... Args>
void BM_op(benchmark::State &state, Args &&...args) {
  auto args_tuple = std::make_tuple(std::move(args)...);
  auto func = std::get<0>(args_tuple);
  size_t N = std::get<1>(args_tuple);
  float min = std::get<2>(args_tuple);
  float max = std::get<3>(args_tuple);

  auto in = hwy::AllocateAligned<float>(N);
  auto out = hwy::AllocateAligned<float>(N);

  fill_float(in.get(), N, min, max);
  for (auto _ : state) {
    func(in.get(), N, out.get());
    benchmark::DoNotOptimize(out);
  }
}

#if HWY_ONCE

#define BENCHMARK_ALL(OP, range_start, range_end)                    \
  BENCHMARK_CAPTURE(BM_op, OP##Hwy, hwy::OP##Hwy, 1 << 12, range_start,     \
                    range_end);                                             \
  BENCHMARK_CAPTURE(BM_op, OP##Translated, hwy::OP##Translated, 1 << 12,    \
                    range_start, range_end);                                \
  BENCHMARK_CAPTURE(BM_op, OP##Sleef, hwy::OP##Sleef, 1 << 12, range_start, \
                    range_end);

#define BENCHMARK_ALL_FAST(OP, range_start, range_end)               \
  BENCHMARK_CAPTURE(BM_op, OP##Hwy, hwy::OP##Hwy, 1 << 12, range_start,     \
                    range_end);                                             \
  BENCHMARK_CAPTURE(BM_op, OP##Translated, hwy::OP##Translated, 1 << 12,    \
                    range_start, range_end);                                \
  BENCHMARK_CAPTURE(BM_op, OP##Sleef, hwy::OP##Sleef, 1 << 12, range_start, \
                    range_end);                                             \
  BENCHMARK_CAPTURE(BM_op, OP##FastTranslated, hwy::OP##FastTranslated,     \
                    1 << 12, range_start, range_end);                       \
  BENCHMARK_CAPTURE(BM_op, OP##FastSleef, hwy::OP##FastSleef, 1 << 12,      \
                    range_start, range_end);

#define BENCHMARK_ALL_FAST_NO_HWY(OP, range_start, range_end)               \
  BENCHMARK_CAPTURE(BM_op, OP##Translated, hwy::OP##Translated, 1 << 12,    \
                    range_start, range_end);                                \
  BENCHMARK_CAPTURE(BM_op, OP##Sleef, hwy::OP##Sleef, 1 << 12, range_start, \
                    range_end);                                             \
  BENCHMARK_CAPTURE(BM_op, OP##FastTranslated, hwy::OP##FastTranslated,     \
                    1 << 12, range_start, range_end);                       \
  BENCHMARK_CAPTURE(BM_op, OP##FastSleef, hwy::OP##FastSleef, 1 << 12,      \
                    range_start, range_end);

BENCHMARK_ALL(Exp, -FLT_MAX, 104);
BENCHMARK_ALL(Expm1, -FLT_MAX, 104);
BENCHMARK_ALL(Log, std::nextafterf(0, INFINITY), FLT_MAX);
BENCHMARK_ALL(Log1p, 0, 1e38);
BENCHMARK_ALL(Log2, std::nextafterf(0, INFINITY), FLT_MAX);

BENCHMARK_ALL_FAST(Sin,  std::nextafterf(-125, INFINITY),  std::nextafterf(125, -INFINITY));
BENCHMARK_ALL_FAST(Sin, -39000, 39000);
BENCHMARK_ALL_FAST(Cos, std::nextafterf(-125, INFINITY),  std::nextafterf(125, -INFINITY));
BENCHMARK_ALL_FAST(Cos, -39000, 39000);
BENCHMARK_ALL_FAST_NO_HWY(Tan,  std::nextafterf(-125, INFINITY),  std::nextafterf(125, -INFINITY));
BENCHMARK_ALL_FAST_NO_HWY(Tan, -39000, 39000);


BENCHMARK_ALL_FAST(Sinh, -88, +88);
BENCHMARK_ALL_FAST_NO_HWY(Cosh, -88, +88);
BENCHMARK_ALL_FAST(Tanh, -FLT_MAX, FLT_MAX);

BENCHMARK_ALL_FAST(Asin, -1, 1);
BENCHMARK_ALL_FAST(Acos, -1, 1);
BENCHMARK_ALL_FAST(Atan, -FLT_MAX, FLT_MAX);

constexpr double SQRT_FLT_MAX = 18446743523953729536.0;

BENCHMARK_ALL(Asinh, -SQRT_FLT_MAX, SQRT_FLT_MAX);
BENCHMARK_ALL(Acosh, 1, SQRT_FLT_MAX);
BENCHMARK_ALL(Atanh, std::nextafterf(-1.0, INFINITY), std::nextafterf(1.0, -INFINITY));

BENCHMARK_MAIN();

#endif
