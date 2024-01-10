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

#include <stdio.h>

#include <algorithm>
#include <cfloat>  // FLT_MAX
#include <cmath>   // std::abs
#include <limits>
#include <memory>

// clang-format off
//#undef HWY_TARGET_INCLUDE
//#define HWY_TARGET_INCLUDE "sleef_test.cc"
//#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/tests/test_util.h"
// clang-format on

#include "src/gen-bindings/sleef-generated.h"
#include "sleef/include/sleefinline_avx2.h"

double ulp_float(double val) {
  if (val == 0) {
    return std::numeric_limits<float>::denorm_min();
  }
  float val1_f = val;
  float val2_f;
  int32_t x;
  hwy::CopySameSize(&val1_f, &x);
  x += val1_f < val ? 1 : -1;
  hwy::CopySameSize(&x, &val2_f);

  return std::abs(((double) val1_f) - ((double) val2_f));
}

double ulp_delta_float(double expected, float actual) {
  if (std::isinf((float)expected) && ((float)expected) == actual) return 0;
  if (std::isnan(expected) && std::isnan(actual)) return 0;
  if (((float)expected) == 0 && actual == 0) return 0;

  return std::abs(expected - (double)actual) / ulp_float(expected);
}

struct accuracy_stat {
  double max_ulp = 0;
  float worst_in, worst_out;
  bool preserve_nan = true;
  char pad_cacheline[64];
};

accuracy_stat merge_stats(const accuracy_stat &a, const accuracy_stat &b) {
  accuracy_stat res;
  if (a.max_ulp > b.max_ulp) {
    res.max_ulp = a.max_ulp;
    res.worst_in = a.worst_in;
    res.worst_out = a.worst_out;
  } else {
    res.max_ulp = b.max_ulp;
    res.worst_in = b.worst_in;
    res.worst_out = b.worst_out;
  }
  res.preserve_nan = a.preserve_nan && b.preserve_nan;
  return res;
}

void update_stat(float in, double expected, float actual, accuracy_stat &stat) {
  if (std::isnan(expected) && !std::isnan(actual)) {
    stat.preserve_nan = false;
  } else {
    double ulp = ulp_delta_float(expected, actual);
    if (ulp > stat.max_ulp) {
      stat.max_ulp = ulp;
      stat.worst_in = in;
      stat.worst_out = actual;
    }
  }
}

HWY_NOINLINE void TestAllFloats1(
    const char *name, double (*ref_func)(double),
    std::initializer_list<
        std::pair<const char *, void (*)(const float *, size_t, float * __restrict__)>>
        func_list,
    float range_start, float range_end) {
  hwy::ThreadPool pool;

  // Compare highway, sleef-native, and translated functions on all floats
  constexpr size_t TASK_SIZE = 1 << 25;   // Count of values per task
  constexpr size_t CHUNK_SIZE = 1 << 8;  // Size of memory per chunk

  size_t func_count = func_list.size();
  std::vector<const char *> func_labels;
  std::vector<void (*)(const float *, size_t, float *)> funcs;
  for (const auto &p : func_list) {
    func_labels.push_back(p.first);
    funcs.push_back(p.second);
  }

  std::unique_ptr<accuracy_stat[]> global_stats = std::make_unique<accuracy_stat[]>(func_count * pool.NumThreads());
  std::unique_ptr<accuracy_stat[]> range_stats = std::make_unique<accuracy_stat[]>(func_count * pool.NumThreads());

  auto run_task = [funcs, &global_stats, &range_stats, TASK_SIZE, CHUNK_SIZE, func_count, range_start, range_end, ref_func](uint32_t task, uint32_t thread) {
    accuracy_stat global_stat, range_stat;
    uint32_t input_int[CHUNK_SIZE];
    float *input = (float *) input_int;
    //std::unique_ptr<float[]> actual = std::make_unique<float[]>(func_count * CHUNK_SIZE);
    float actual[3 * CHUNK_SIZE];

    accuracy_stat global_stats_thread[3], range_stats_thread[3];
    //std::unique_ptr<accuracy_stat[]> global_stats_thread = std::make_unique<accuracy_stat[]>(func_count);
    //std::unique_ptr<accuracy_stat[]> range_stats_thread = std::make_unique<accuracy_stat[]>(func_count);

    for (size_t i = task * TASK_SIZE; i < (task + 1) * TASK_SIZE;
         i += CHUNK_SIZE) {
      for (size_t j = 0; j < CHUNK_SIZE; j++) {
        input_int[j] = i + j;
      }
      for (size_t f = 0; f < func_count; f++) {
        funcs[f](input, CHUNK_SIZE, actual + f * CHUNK_SIZE);
      }

      for (size_t j = 0; j < CHUNK_SIZE; j++) {
        double expected = ref_func(input[j]);
        for (size_t f = 0; f < func_count; f++) {
          update_stat(input[j], expected, actual[j + f * CHUNK_SIZE], global_stats_thread[f]);
          if (range_start <= input[j] && range_end >= input[j]) {
            update_stat(input[j], expected, actual[j + f * CHUNK_SIZE],
                        range_stats_thread[f]);
          }
        }
      }
    }
    
    for (size_t f = 0; f < func_count; f++) {
      global_stats[thread*func_count + f] = merge_stats(global_stats_thread[f], global_stats[thread*func_count + f]);
      range_stats[thread*func_count + f] = merge_stats(range_stats_thread[f], range_stats[thread*func_count + f]);

    }
  };
  pool.Run(0, UINT32_MAX / TASK_SIZE, hwy::ThreadPool::NoInit, run_task);

  printf("##########\n# %s\n##########\n", name);
  printf(
      "Global stats:\n"
      "%-10s%10s%15s%15s%15s\n"
      "----------------------------------------------------------------\n",
      "Method", "Max ULP", "Preserve NaN", "Worst input", "Worst output");

  for (size_t f = 0; f < func_count; f++) {
    accuracy_stat result;
    for (size_t t = 0; t < pool.NumThreads(); t++) {
      result = merge_stats(result, global_stats[f + t * func_count]);
    }
    printf("%-10s%10.3g%15s%15a%15a\n", func_labels[f], result.max_ulp,
           result.preserve_nan ? "true" : "false", result.worst_in,
           result.worst_out);
  }

  printf(
      "\nIn-range stats: [%f,%f]\n"
      "%-10s%10s%15s%15s%15s\n"
      "----------------------------------------------------------------\n",
      range_start, range_end, "Method", "Max ULP", "Preserve NaN",
      "Worst input", "Worst output");

  for (size_t f = 0; f < func_count; f++) {
    accuracy_stat result;
    for (size_t t = 0; t < pool.NumThreads(); t++) {
      result = merge_stats(result, range_stats[f + t * func_count]);
    }
    printf("%-10s%10.3g%15s%15a%15a\n", func_labels[f], result.max_ulp,
           result.preserve_nan ? "true" : "false", result.worst_in,
           result.worst_out);
  }
  printf("\n");
}

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

#define WRAP_OP1(NAME, FN)                                        \
  void NAME(const float *in, size_t n, float *__restrict__ out) { \
    using D = ScalableTag<float>;                                 \
    D d;                                                          \
    Vec<D> v;                                                     \
    size_t lanes = Lanes(d);                                      \
    for (size_t i = 0; i < n; i += lanes) {                       \
      v = LoadU(d, in + i);                                       \
      v = FN(d, v);                                               \
      StoreU(v, d, out + i);                                      \
    }                                                             \
  }

WRAP_OP1(HwyExp, Exp);
WRAP_OP1(MyExp, sleef::Exp);

WRAP_OP1(HwyExpm1, Expm1);
WRAP_OP1(MyExpm1, sleef::Expm1);

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

namespace hwy {

#define DISPATCH_OP1(FN)                                        \
  void FN(const float *in, size_t n, float *__restrict__ out) { \
    HWY_STATIC_DISPATCH(FN)(in, n, out);                        \
  }
DISPATCH_OP1(HwyExp)
DISPATCH_OP1(MyExp)

DISPATCH_OP1(HwyExpm1)
DISPATCH_OP1(MyExpm1)

}  // namespace hwy

#define WRAP_SLEEF(NAME, FN)                                          \
  void NAME(const float *in, size_t n, float *__restrict__ out) { \
    __m256 v;                                                         \
    for (size_t i = 0; i < n; i += 8) {                               \
      v = _mm256_loadu_ps(in + i);                                    \
      v = FN(v);                                     \
      _mm256_storeu_ps(out + i, v);                                   \
    }                                                                 \
  }
WRAP_SLEEF(SleefExp, Sleef_expf8_u10avx2)
WRAP_SLEEF(SleefExpm1, Sleef_expm1f8_u10avx2)

int main() {
  // float in[] = {-1, -0.0, 0, 0.5, 1, 2, 3, 4};
  // float out[8];
  // for (int i = 0; i < 8; i++) {
  //   printf("expm1(%f) = %f\n", (double) in[i], expm1(in[i]));
  // }
  // hwy::HwyExpm1(in, 8, out);
  // for (int i = 0; i < 8; i++) {
  //   printf("HwyExpm1(%f) = %f (%f ulp)\n", (double) in[i], out[i], ulp_delta_float(expm1(in[i]), out[i]));
  // }
  // SleefExpm1(in, 8, out);
  // for (int i = 0; i < 8; i++) {
  //   printf("SleefExpm1(%f) = %f (%f ulp)\n", (double) in[i], out[i], ulp_delta_float(expm1(in[i]), out[i]));
  // }
  // hwy::MyExpm1(in, 8, out);
  // for (int i = 0; i < 8; i++) {
  //   printf("MyExpm1(%f) = %f (%f ulp)\n", (double) in[i], out[i], ulp_delta_float(expm1(in[i]), out[i]));
  // }

  // TestAllFloats1(
  //     "expm1", expm1,
  //     {{"Hwy", hwy::HwyExpm1}, {"Sleef", SleefExpm1}, {"Translated", hwy::MyExpm1}},
  //     -INFINITY, 104.0);
  TestAllFloats1(
      "exp", exp,
      {{"Hwy", hwy::HwyExp}, {"Sleef", SleefExp}, {"Translated", hwy::MyExp}},
      -INFINITY, 104.0);

}
#endif
