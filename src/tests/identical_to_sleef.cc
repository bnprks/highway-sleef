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
#include <iostream>

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

template <typename real, typename integer>
void FindMismatches(const char *name,
                    void (*sleef_fun)(const real *, size_t,
                                      real *__restrict__),
                    void (*translate_fun)(const real *, size_t,
                                          real *__restrict__)) {
  hwy::ThreadPool pool(48);

  // Compare highway, sleef-native, and translated functions on all floats
  constexpr size_t TASK_SIZE = 1 << 24;   // Count of values per task
  constexpr size_t CHUNK_SIZE = 1 << 10;  // Size of memory per chunk

  struct mismatch {
    integer input, sleef_out, translate_out;
    integer mismatch_type =
        0;  // 0 = no mismatch, 1 = nan mismatch, 2 = value mismatch
    char _cache_line_padding[64];
  };
  std::vector<struct mismatch> mismatches(pool.NumThreads());

  std::atomic<bool> quit_early;
  quit_early = false;

  auto run_task = [=, &mismatches, &quit_early](uint32_t task,
                                                uint32_t thread) {
    auto out1 = hwy::AllocateAligned<real>(CHUNK_SIZE);
    auto out2 = hwy::AllocateAligned<real>(CHUNK_SIZE);
    auto input = hwy::AllocateAligned<real>(CHUNK_SIZE);
    bool mismatched_nan = false;

    if (quit_early) return;

    for (size_t i = task * TASK_SIZE; i < (task + 1) * TASK_SIZE;
         i += CHUNK_SIZE) {
      for (size_t j = 0; j < CHUNK_SIZE; j++) {
        input[j] = hwy::BitCastScalar<real>(static_cast<integer>(i + j));
      }
      sleef_fun(input.get(), CHUNK_SIZE, out1.get());
      translate_fun(input.get(), CHUNK_SIZE, out2.get());
      for (size_t j = 0; j < CHUNK_SIZE; j++) {
        if (hwy::BitCastScalar<integer>(out1[j]) !=
            hwy::BitCastScalar<integer>(out2[j])) {
          if (std::isnan(out1[j]) && std::isnan(out2[j])) {
            if (!mismatched_nan) {
              mismatches[thread].mismatch_type = 1;
              mismatches[thread].input = hwy::BitCastScalar<integer>(input[j]);
              mismatches[thread].sleef_out =
                  hwy::BitCastScalar<integer>(out1[j]);
              mismatches[thread].translate_out =
                  hwy::BitCastScalar<integer>(out2[j]);
            }
            mismatched_nan = true;
          } else {
            mismatches[thread].mismatch_type = 2;
            mismatches[thread].input = hwy::BitCastScalar<integer>(input[j]);
            mismatches[thread].sleef_out =
                hwy::BitCastScalar<integer>(out1[j]);
            mismatches[thread].translate_out =
                hwy::BitCastScalar<integer>(out2[j]);
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
          name, hwy::BitCastScalar<real>(m.input), m.input,
          hwy::BitCastScalar<real>(m.sleef_out), m.sleef_out,
          hwy::BitCastScalar<real>(m.translate_out), m.translate_out);
      return;
    }
  }
  bool mismatched_nan = false;
  for (const auto &m : mismatches) {
    if (m.mismatch_type == 1) {
      printf(
          "%s: found nan-mismatch: in = %f (0x%x), sleef = %f (0x%x) vs. "
          "translated = %f (0x%x)\n",
          name, hwy::BitCastScalar<real>(m.input), m.input,
          hwy::BitCastScalar<real>(m.sleef_out), m.sleef_out,
          hwy::BitCastScalar<real>(m.translate_out), m.translate_out);
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


#define FIND_MISMATCHES_HELPER_SINGLE(OP) \
  FindMismatches<float, uint32_t>(#OP "f", hwy::OP##f##Translated, hwy::OP##f##Sleef)

#define FIND_MISMATCHES_HELPER_DOUBLE(OP) \
  FindMismatches<double, uint64_t>(#OP "d", hwy::OP##d##Translated, hwy::OP##d##Sleef)

int main() {
  FIND_MISMATCHES_HELPER_SINGLE(Exp);
  FIND_MISMATCHES_HELPER_SINGLE(Expm1);
  FIND_MISMATCHES_HELPER_SINGLE(Log);
  FIND_MISMATCHES_HELPER_SINGLE(Log1p);
  FIND_MISMATCHES_HELPER_SINGLE(Log2);

  FIND_MISMATCHES_HELPER_DOUBLE(Exp);
  FIND_MISMATCHES_HELPER_DOUBLE(Expm1);
  FIND_MISMATCHES_HELPER_DOUBLE(Log);
  FIND_MISMATCHES_HELPER_DOUBLE(Log1p);
  FIND_MISMATCHES_HELPER_DOUBLE(Log2);


  FIND_MISMATCHES_HELPER_SINGLE(Sin);
  FIND_MISMATCHES_HELPER_SINGLE(Cos);
  FIND_MISMATCHES_HELPER_SINGLE(Tan);
  FIND_MISMATCHES_HELPER_SINGLE(SinFast);
  FIND_MISMATCHES_HELPER_SINGLE(CosFast);
  FIND_MISMATCHES_HELPER_SINGLE(TanFast);

  FIND_MISMATCHES_HELPER_DOUBLE(Sin);
  FIND_MISMATCHES_HELPER_DOUBLE(Cos);
  FIND_MISMATCHES_HELPER_DOUBLE(Tan);
  FIND_MISMATCHES_HELPER_DOUBLE(SinFast);
  FIND_MISMATCHES_HELPER_DOUBLE(CosFast);
  FIND_MISMATCHES_HELPER_DOUBLE(TanFast);

  FIND_MISMATCHES_HELPER_SINGLE(Sinh);
  FIND_MISMATCHES_HELPER_SINGLE(Cosh);
  FIND_MISMATCHES_HELPER_SINGLE(Tanh);
  FIND_MISMATCHES_HELPER_SINGLE(SinhFast);
  FIND_MISMATCHES_HELPER_SINGLE(CoshFast);
  FIND_MISMATCHES_HELPER_SINGLE(TanhFast);

  FIND_MISMATCHES_HELPER_DOUBLE(Sinh);
  FIND_MISMATCHES_HELPER_DOUBLE(Cosh);
  FIND_MISMATCHES_HELPER_DOUBLE(Tanh);
  FIND_MISMATCHES_HELPER_DOUBLE(SinhFast);
  FIND_MISMATCHES_HELPER_DOUBLE(CoshFast);
  FIND_MISMATCHES_HELPER_DOUBLE(TanhFast);


  FIND_MISMATCHES_HELPER_SINGLE(Asin);
  FIND_MISMATCHES_HELPER_SINGLE(Acos);
  FIND_MISMATCHES_HELPER_SINGLE(Atan);
  FIND_MISMATCHES_HELPER_SINGLE(AsinFast);
  FIND_MISMATCHES_HELPER_SINGLE(AcosFast);
  FIND_MISMATCHES_HELPER_SINGLE(AtanFast);

  FIND_MISMATCHES_HELPER_SINGLE(Asinh);
  FIND_MISMATCHES_HELPER_SINGLE(Acosh);
  FIND_MISMATCHES_HELPER_SINGLE(Atanh);

}
#endif
