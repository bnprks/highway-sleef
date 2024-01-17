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
#include <atomic>
#include <cfloat>  // FLT_MAX
#include <cmath>   // std::abs
#include <limits>
#include <memory>

#include "define_dispatchers.h"

// clang-format off
//#undef HWY_TARGET_INCLUDE
//#define HWY_TARGET_INCLUDE "sleef_test.cc"
//#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/tests/test_util.h"
// clang-format on

HWY_INLINE double ulp_float(double val) {
  if (val == 0) {
    return std::numeric_limits<float>::denorm_min();
  }
  float val1_f = val;
  float val2_f;
  int32_t x;
  hwy::CopySameSize(&val1_f, &x);
  x += val1_f < val ? 1 : -1;
  hwy::CopySameSize(&x, &val2_f);

  return std::abs(((double)val1_f) - ((double)val2_f));
}

HWY_INLINE double ulp_delta_float(double expected, float actual) {
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

HWY_INLINE void update_stat(float in, double expected, float actual,
                            accuracy_stat &stat) {
  if (std::isnan(expected) && !std::isnan(actual)) {
    stat.preserve_nan = false;
  } else {
    double ulp = ulp_delta_float(expected, actual);
    if (HWY_UNLIKELY(ulp > stat.max_ulp)) {
      stat.max_ulp = ulp;
      stat.worst_in = in;
      stat.worst_out = actual;
    }
  }
}

HWY_NOINLINE void AllFloatULP(
    const char *name, double (*ref_func)(double),
    std::initializer_list<std::pair<
        const char *, void (*)(const float *, size_t, float *__restrict__)>>
        func_list,
    float range_start, float range_end) {
  hwy::ThreadPool pool(4);

  // Compare highway, sleef-native, and translated functions on all floats
  constexpr size_t TASK_SIZE = 1 << 25;  // Count of values per task
  constexpr size_t CHUNK_SIZE = 1 << 8;  // Size of memory per chunk

  size_t func_count = func_list.size();
  std::vector<const char *> func_labels;
  std::vector<void (*)(const float *, size_t, float *)> funcs;
  for (const auto &p : func_list) {
    func_labels.push_back(p.first);
    funcs.push_back(p.second);
  }

  std::unique_ptr<accuracy_stat[]> global_stats =
      std::make_unique<accuracy_stat[]>(func_count * pool.NumThreads());
  std::unique_ptr<accuracy_stat[]> range_stats =
      std::make_unique<accuracy_stat[]>(func_count * pool.NumThreads());


  auto run_task = [=, &global_stats, &range_stats](uint32_t task, uint32_t thread) {
    accuracy_stat global_stat, range_stat;

    auto input = hwy::AllocateAligned<float>(CHUNK_SIZE);
    
    std::vector<hwy::AlignedFreeUniquePtr<float []>> outputs;
    for (int i = 0; i < func_count; i++) {
      outputs.push_back(hwy::AllocateAligned<float>(CHUNK_SIZE));
    }

    bool identical = true;

    accuracy_stat global_stats_thread[3], range_stats_thread[3];


    for (size_t i = task * TASK_SIZE; i < (task + 1) * TASK_SIZE;
         i += CHUNK_SIZE) {
      for (size_t j = 0; j < CHUNK_SIZE; j++) {
        input[j] = hwy::BitCastScalar<float>(static_cast<uint32_t>(i + j));
      }
      for (size_t f = 0; f < func_count; f++) {
        funcs[f](input.get(), CHUNK_SIZE, outputs[f].get());
      }

      for (size_t j = 0; j < CHUNK_SIZE; j++) {
        double expected = ref_func(input[j]);
        for (size_t f = 0; f < func_count; f++) {
          update_stat(input[j], expected, outputs[f][j],
                      global_stats_thread[f]);
          if (range_start <= input[j] && range_end >= input[j]) {
            update_stat(input[j], expected, outputs[f][j],
                        range_stats_thread[f]);
          }
        }
      }
    }

    for (size_t f = 0; f < func_count; f++) {
      global_stats[thread * func_count + f] = merge_stats(
          global_stats_thread[f], global_stats[thread * func_count + f]);
      range_stats[thread * func_count + f] = merge_stats(
          range_stats_thread[f], range_stats[thread * func_count + f]);
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

#if HWY_ONCE


#define ALL_FLOAT_ULP_HELPER(OP, ref, range_start, range_end) \
  AllFloatULP(#OP, ref, {{"hwy", hwy::OP##Hwy}, {"translated", hwy::OP##Translated}}, range_start, range_end)

#define ALL_FLOAT_ULP_FAST_HELPER(OP, ref, range_start, range_end) \
  AllFloatULP(#OP, ref, {{"hwy", hwy::OP##Hwy}, {"translated", hwy::OP##Translated}, {"translated_fast", hwy::OP##FastTranslated}}, range_start, range_end)

int main() {
  ALL_FLOAT_ULP_HELPER(Exp, std::exp, -FLT_MAX, 104);
  ALL_FLOAT_ULP_HELPER(Expm1, std::expm1, -FLT_MAX, 104);
  ALL_FLOAT_ULP_HELPER(Log, std::log, std::nextafterf(0, INFINITY), FLT_MAX);
  ALL_FLOAT_ULP_HELPER(Log1p, std::log1p, 0, FLT_MAX);
  ALL_FLOAT_ULP_HELPER(Log1p, std::log1p, std::nextafterf(-1, INFINITY), 1e38);
  ALL_FLOAT_ULP_HELPER(Log2, std::log2, std::nextafterf(0, INFINITY), FLT_MAX);

  ALL_FLOAT_ULP_FAST_HELPER(Sin, std::sin, -39000, 39000);
  ALL_FLOAT_ULP_FAST_HELPER(Cos, std::cos, -39000, 39000);
  AllFloatULP("Tan", std::tan, {{"translated", hwy::TanTranslated}, {"translated_fast", hwy::TanFastTranslated}}, -39000, 39000);

  ALL_FLOAT_ULP_FAST_HELPER(Sinh, std::sinh, -88.7228, +88.7228);
  ALL_FLOAT_ULP_FAST_HELPER(Sinh, std::sinh, -89, +89);
  ALL_FLOAT_ULP_FAST_HELPER(Sinh, std::sinh, -88, +88);
  AllFloatULP("Cosh", std::cosh, {{"translated", hwy::CoshTranslated}, {"translated_fast", hwy::CoshFastTranslated}}, -89, +89);
  AllFloatULP("Cosh", std::cosh, {{"translated", hwy::CoshTranslated}, {"translated_fast", hwy::CoshFastTranslated}}, -88, +88);
  ALL_FLOAT_ULP_FAST_HELPER(Tanh, std::tanh, -FLT_MAX, FLT_MAX);

  ALL_FLOAT_ULP_FAST_HELPER(Asin, std::asin, -1, 1);
  ALL_FLOAT_ULP_FAST_HELPER(Acos, std::acos, -1, 1);
  ALL_FLOAT_ULP_FAST_HELPER(Atan, std::atan, -FLT_MAX, FLT_MAX);

  constexpr double SQRT_FLT_MAX = 18446743523953729536.0;

  ALL_FLOAT_ULP_HELPER(Asinh, std::asinh, -FLT_MAX, FLT_MAX);
  ALL_FLOAT_ULP_HELPER(Asinh, std::asinh, -SQRT_FLT_MAX, SQRT_FLT_MAX);
  ALL_FLOAT_ULP_HELPER(Acosh, std::acosh, 1, FLT_MAX);
  ALL_FLOAT_ULP_HELPER(Acosh, std::acosh, 1, SQRT_FLT_MAX);

  ALL_FLOAT_ULP_HELPER(Atanh, std::atanh, std::nextafterf(-1.0, INFINITY), std::nextafterf(1.0, -INFINITY));
}
#endif
