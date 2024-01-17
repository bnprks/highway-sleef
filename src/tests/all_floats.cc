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

// clang-format off
//#undef HWY_TARGET_INCLUDE
//#define HWY_TARGET_INCLUDE "sleef_test.cc"
//#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/tests/test_util.h"
// clang-format on
#include "sleef.h"
#include "src/gen-bindings/sleef-generated.h"

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
    if (ulp > stat.max_ulp) {
      stat.max_ulp = ulp;
      stat.worst_in = in;
      stat.worst_out = actual;
    }
  }
}

void SingleMismatch(const char *name,
                    void (*f1)(const float *, size_t, float *__restrict__),
                    void (*f2)(const float *, size_t, float *__restrict__),
                    float val) {
  float input[8];
  float out1[8];
  float out2[8];
  for (int i = 0; i < 8; i++) {
    input[i] = val;
  }
  printf("Running first:\n");
  f1(input, 8, out1);
  printf("Running second:\n");
  f2(input, 8, out2);
  printf("in=%a, out1=%a, out2=%a\n", input[0], out1[0], out2[0]);
  printf("in=%d, out1=%d, out2=%d\n", ((uint32_t *)input)[0],
         ((uint32_t *)out1)[0], ((uint32_t *)out2)[0]);
}

void FindMismatches(const char *name,
                    void (*sleef_fun)(const float *, size_t,
                                      float *__restrict__),
                    void (*translate_fun)(const float *, size_t,
                                          float *__restrict__)) {
  hwy::ThreadPool pool(4);

  // Compare highway, sleef-native, and translated functions on all floats
  constexpr size_t TASK_SIZE = 1 << 24;   // Count of values per task
  constexpr size_t CHUNK_SIZE = 1 << 10;  // Size of memory per chunk

  struct mismatch {
    uint32_t input, sleef_out, translate_out;
    uint32_t mismatch_type =
        0;  // 0 = no mismatch, 1 = nan mismatch, 2 = value mismatch
    char _cache_line_padding[64];
  };
  std::vector<struct mismatch> mismatches(pool.NumThreads());

  std::atomic<bool> quit_early;
  quit_early = false;

  auto run_task = [=, &mismatches, &quit_early](uint32_t task,
                                                uint32_t thread) {
    auto out1 = hwy::AllocateAligned<float>(CHUNK_SIZE);
    auto out2 = hwy::AllocateAligned<float>(CHUNK_SIZE);
    auto input = hwy::AllocateAligned<float>(CHUNK_SIZE);
    bool mismatched_nan = false;

    if (quit_early) return;

    for (size_t i = task * TASK_SIZE; i < (task + 1) * TASK_SIZE;
         i += CHUNK_SIZE) {
      for (size_t j = 0; j < CHUNK_SIZE; j++) {
        input[j] = hwy::BitCastScalar<float>(static_cast<uint32_t>(i + j));
      }
      sleef_fun(input.get(), CHUNK_SIZE, out1.get());
      translate_fun(input.get(), CHUNK_SIZE, out2.get());
      for (size_t j = 0; j < CHUNK_SIZE; j++) {
        if (hwy::BitCastScalar<uint32_t>(out1[j]) !=
            hwy::BitCastScalar<uint32_t>(out2[j])) {
          if (std::isnan(out1[j]) && std::isnan(out2[j])) {
            if (!mismatched_nan) {
              mismatches[thread].mismatch_type = 1;
              mismatches[thread].input = hwy::BitCastScalar<uint32_t>(input[j]);
              mismatches[thread].sleef_out =
                  hwy::BitCastScalar<uint32_t>(out1[j]);
              mismatches[thread].translate_out =
                  hwy::BitCastScalar<uint32_t>(out2[j]);
            }
            mismatched_nan = true;
          } else {
            mismatches[thread].mismatch_type = 2;
            mismatches[thread].input = hwy::BitCastScalar<uint32_t>(input[j]);
            mismatches[thread].sleef_out =
                hwy::BitCastScalar<uint32_t>(out1[j]);
            mismatches[thread].translate_out =
                hwy::BitCastScalar<uint32_t>(out2[j]);
            quit_early = true;
            return;
          }
        }
      }
    }
  };

  pool.Run(0, UINT32_MAX / TASK_SIZE, hwy::ThreadPool::NoInit, run_task);

  // Check for value mismatches
  for (const auto &m : mismatches) {
    if (m.mismatch_type == 2) {
      printf(
          "%s: found mismatch: in = %f (0x%x), sleef = %f (0x%x) vs. "
          "translated = %f (0x%x)\n",
          name, hwy::BitCastScalar<float>(m.input), m.input,
          hwy::BitCastScalar<float>(m.sleef_out), m.sleef_out,
          hwy::BitCastScalar<float>(m.translate_out), m.translate_out);
      return;
    }
  }
  bool mismatched_nan = false;
  for (const auto &m : mismatches) {
    if (m.mismatch_type == 1) {
      printf(
          "%s: found nan-mismatch: in = %f (0x%x), sleef = %f (0x%x) vs. "
          "translated = %f (0x%x)\n",
          name, hwy::BitCastScalar<float>(m.input), m.input,
          hwy::BitCastScalar<float>(m.sleef_out), m.sleef_out,
          hwy::BitCastScalar<float>(m.translate_out), m.translate_out);
      mismatched_nan = true;
      break;
    }
  }

  if (mismatched_nan) {
    printf("%s: all non-nan outputs match\n", name);
  } else {
    printf("%s: all outputs match\n", name);
  }
}

HWY_NOINLINE void TestAllFloats1(
    const char *name, double (*ref_func)(double),
    std::initializer_list<std::pair<
        const char *, void (*)(const float *, size_t, float *__restrict__)>>
        func_list,
    float range_start, float range_end) {
  hwy::ThreadPool pool;

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
  std::unique_ptr<bool[]> identical_to_sleef =
      std::make_unique<bool[]>(pool.NumThreads());
  for (size_t t = 0; t < pool.NumThreads(); t++) {
    identical_to_sleef[t] = true;
  }

  auto run_task = [funcs, &global_stats, &range_stats, &identical_to_sleef,
                   TASK_SIZE, CHUNK_SIZE, func_count, range_start, range_end,
                   ref_func](uint32_t task, uint32_t thread) {
    accuracy_stat global_stat, range_stat;
    uint32_t input_int[CHUNK_SIZE];
    float *input = (float *)input_int;
    // std::unique_ptr<float[]> actual = std::make_unique<float[]>(func_count
    // * CHUNK_SIZE);
    float actual[3 * CHUNK_SIZE];
    bool identical = true;

    accuracy_stat global_stats_thread[3], range_stats_thread[3];
    // std::unique_ptr<accuracy_stat[]> global_stats_thread =
    // std::make_unique<accuracy_stat[]>(func_count);
    // std::unique_ptr<accuracy_stat[]> range_stats_thread =
    // std::make_unique<accuracy_stat[]>(func_count);

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
          update_stat(input[j], expected, actual[j + f * CHUNK_SIZE],
                      global_stats_thread[f]);
          if (range_start <= input[j] && range_end >= input[j]) {
            update_stat(input[j], expected, actual[j + f * CHUNK_SIZE],
                        range_stats_thread[f]);
          }
        }
        identical =
            identical && (actual[j + CHUNK_SIZE] == actual[j + 2 * CHUNK_SIZE]);
      }
    }

    for (size_t f = 0; f < func_count; f++) {
      global_stats[thread * func_count + f] = merge_stats(
          global_stats_thread[f], global_stats[thread * func_count + f]);
      range_stats[thread * func_count + f] = merge_stats(
          range_stats_thread[f], range_stats[thread * func_count + f]);
      identical_to_sleef[thread] = identical_to_sleef[thread] && identical;
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
  bool identical = true;
  for (size_t t = 0; t < pool.NumThreads(); t++) {
    identical = identical && identical_to_sleef[t];
  }
  printf("Last two identical: %s\n", identical ? "true" : "false");

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

// Register an AVX2 highway wrapper for a Sleef function
#define SLEEF_TO_HWY_F8(NAME, FN)                \
  HWY_BEFORE_NAMESPACE();                        \
  namespace hwy {                                \
  namespace N_AVX2 {                             \
  template <class D>                             \
  HWY_INLINE Vec<D> NAME(const D df, Vec<D> d) { \
    return Vec256<float>{FN(d.raw)};             \
  }                                              \
  }                                              \
  }                                              \
  HWY_AFTER_NAMESPACE();

#define WRAP_OP1(NAME, FN)                                        \
  HWY_BEFORE_NAMESPACE();                                         \
  namespace hwy {                                                 \
  namespace HWY_NAMESPACE {                                       \
  void NAME(const float *in, size_t n, float *__restrict__ out) { \
    using D = ScalableTag<float>;                                 \
    D d;                                                          \
    Vec<D> v;                                                     \
    size_t lanes = Lanes(d);                                      \
    for (size_t i = 0; i < n; i += lanes) {                       \
      v = Load(d, in + i);                                        \
      v = FN(d, v);                                               \
      Store(v, d, out + i);                                       \
    }                                                             \
  }                                                               \
  }                                                               \
  }                                                               \
  HWY_AFTER_NAMESPACE();

#define DISPATCH_OP1(NAME, FN)                                                 \
  namespace hwy {                                                              \
  HWY_NOINLINE void NAME(const float *in, size_t n, float *__restrict__ out) { \
    HWY_STATIC_DISPATCH(NAME)(in, n, out);                                     \
  }                                                                            \
  }

#define DISPATCH_AND_WRAP_OP1(NAME, FN) \
  WRAP_OP1(NAME, FN)                    \
  DISPATCH_OP1(NAME, FN)

// Define dispatch names based on common naming expectations for operations
#define DISPATCH_ALL(OP, SLEEF_NAME)               \
  DISPATCH_AND_WRAP_OP1(OP##Hwy, OP)               \
  DISPATCH_AND_WRAP_OP1(OP##Translated, sleef::OP) \
  SLEEF_TO_HWY_F8(OP##Sleef, SLEEF_NAME)           \
  DISPATCH_AND_WRAP_OP1(OP##Sleef, OP##Sleef)

#define DISPATCH_ALL_SKIP_HWY(OP, SLEEF_NAME) \
  DISPATCH_AND_WRAP_OP1(OP##Translated, sleef::OP) \
  SLEEF_TO_HWY_F8(OP##Sleef, SLEEF_NAME)           \
  DISPATCH_AND_WRAP_OP1(OP##Sleef, OP##Sleef)

// Define dispatch names for low-precision variants which also have a high
// precision variant
#define DISPATCH_ALL_LOW_PRECISION(OP, SLEEF_NAME)           \
  DISPATCH_AND_WRAP_OP1(OP##FastTranslated, sleef::OP##Fast) \
  SLEEF_TO_HWY_F8(OP##FastSleef, SLEEF_NAME)                 \
  DISPATCH_AND_WRAP_OP1(OP##FastSleef, OP##FastSleef)

DISPATCH_ALL(Exp, Sleef_finz_expf8_u10avx2)
DISPATCH_ALL(Expm1, Sleef_finz_expm1f8_u10avx2)
DISPATCH_ALL(Log, Sleef_finz_logf8_u10avx2)
DISPATCH_ALL(Log1p, Sleef_finz_log1pf8_u10avx2)
DISPATCH_ALL(Log2, Sleef_finz_log2f8_u10avx2)

DISPATCH_ALL(Sin, Sleef_finz_sinf8_u10avx2)
DISPATCH_ALL(Cos, Sleef_finz_cosf8_u10avx2)
DISPATCH_ALL_SKIP_HWY(Tan, Sleef_finz_tanf8_u10avx2)
DISPATCH_ALL_LOW_PRECISION(Sin, Sleef_finz_sinf8_u35avx2)
DISPATCH_ALL_LOW_PRECISION(Cos, Sleef_finz_cosf8_u35avx2)
DISPATCH_ALL_LOW_PRECISION(Tan, Sleef_finz_tanf8_u35avx2)

DISPATCH_ALL(Sinh, Sleef_finz_sinhf8_u10avx2)
DISPATCH_ALL_SKIP_HWY(Cosh, Sleef_finz_coshf8_u10avx2)
DISPATCH_ALL(Tanh, Sleef_finz_tanhf8_u10avx2)
DISPATCH_ALL_LOW_PRECISION(Sinh, Sleef_finz_sinhf8_u35avx2)
DISPATCH_ALL_LOW_PRECISION(Cosh, Sleef_finz_coshf8_u35avx2)
DISPATCH_ALL_LOW_PRECISION(Tanh, Sleef_finz_tanhf8_u35avx2)


DISPATCH_ALL(Asin, Sleef_finz_asinf8_u10avx2)
DISPATCH_ALL(Acos, Sleef_finz_acosf8_u10avx2)
DISPATCH_ALL(Atan, Sleef_finz_atanf8_u10avx2)
DISPATCH_ALL_LOW_PRECISION(Asin, Sleef_finz_asinf8_u35avx2)
DISPATCH_ALL_LOW_PRECISION(Acos, Sleef_finz_acosf8_u35avx2)
DISPATCH_ALL_LOW_PRECISION(Atan, Sleef_finz_atanf8_u35avx2)

DISPATCH_ALL(Asinh, Sleef_finz_asinhf8_u10avx2)
DISPATCH_ALL(Acosh, Sleef_finz_acoshf8_u10avx2)
DISPATCH_ALL(Atanh, Sleef_finz_atanhf8_u10avx2)

#define FIND_MISMATCHES_HELPER(OP) \
  FindMismatches(#OP, hwy::OP##Translated, hwy::OP##Sleef)

int main() {
  FIND_MISMATCHES_HELPER(Exp);
  FIND_MISMATCHES_HELPER(Expm1);
  FIND_MISMATCHES_HELPER(Log);
  FIND_MISMATCHES_HELPER(Log1p);
  FIND_MISMATCHES_HELPER(Log2);

  FIND_MISMATCHES_HELPER(Sin);
  FIND_MISMATCHES_HELPER(Cos);
  FIND_MISMATCHES_HELPER(Tan);
  FIND_MISMATCHES_HELPER(SinFast);
  FIND_MISMATCHES_HELPER(CosFast);
  FIND_MISMATCHES_HELPER(TanFast);

  FIND_MISMATCHES_HELPER(Sinh);
  FIND_MISMATCHES_HELPER(Cosh);
  FIND_MISMATCHES_HELPER(Tanh);
  FIND_MISMATCHES_HELPER(SinhFast);
  FIND_MISMATCHES_HELPER(CoshFast);
  FIND_MISMATCHES_HELPER(TanhFast);


  FIND_MISMATCHES_HELPER(Asin);
  FIND_MISMATCHES_HELPER(Acos);
  FIND_MISMATCHES_HELPER(Atan);
  FIND_MISMATCHES_HELPER(AsinFast);
  FIND_MISMATCHES_HELPER(AcosFast);
  FIND_MISMATCHES_HELPER(AtanFast);

  FIND_MISMATCHES_HELPER(Asinh);
  FIND_MISMATCHES_HELPER(Acosh);
  FIND_MISMATCHES_HELPER(Atanh);
}
#endif
