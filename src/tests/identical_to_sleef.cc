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

#if HWY_ONCE


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
