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
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "src/tests/mismatch_test.cc"
#include "hwy/foreach_target.h"  // IWYU pragma: keep
#include "hwy/highway.h"
#include "hwy/aligned_allocator.h"
#include "hwy/contrib/math/math-inl.h"
#include "hwy/contrib/thread_pool/thread_pool.h"
#include "hwy/tests/test_util.h"
// clang-format on

//#include "define_dispatchers.h"
#include "sleef.h"
#include "src/gen-bindings/sleef-generated.h"

// void SingleMismatch(const char *name,
//                     void (*f1)(const float *, size_t, float *__restrict__),
//                     void (*f2)(const float *, size_t, float *__restrict__),
//                     float val) {
//   auto input = hwy::AllocateAligned<float>(8);
//   auto out1 = hwy::AllocateAligned<float>(8);
//   auto out2 = hwy::AllocateAligned<float>(8);
//   for (int i = 0; i < 8; i++) {
//     input[i] = val;
//   }
//   printf("Running first:\n");
//   f1(input.get(), 8, out1.get());
//   printf("Running second:\n");
//   f2(input.get(), 8, out2.get());
//   printf("in=%a, out1=%a, out2=%a\n", input[0], out1[0], out2[0]);
//   printf("in=%d, out1=%d, out2=%d\n", ((uint32_t *)input.get())[0],
//          ((uint32_t *)out1.get())[0], ((uint32_t *)out2.get())[0]);
// }

#if HWY_ONCE

int main() {
    // SingleMismatch("log1p", hwy::Log1pSleef, hwy::Log1pTranslated, 0x1.2ced34p+126);
    // SingleMismatch("log1p", hwy::Log1pHwy, hwy::Log1pTranslated, 0x1.2ced34p+126);
    // printf("Sleef scalar result: %a\n", Sleef_log1pf1_u10(0x1.2ced34p+126));
    // SingleMismatch("log1p", hwy::Log1pSleef, hwy::Log1pTranslated, 0x1.7ffffep+127);
    // SingleMismatch("log1p", hwy::Log1pHwy, hwy::Log1pTranslated, 0x1.7ffffep+127);
    // printf("Sleef scalar result: %a\n", Sleef_log1pf1_u10(0x1.7ffffep+127));

    // printf("Correct result for %a: %a\n", 0x1.7ffffep+127, std::log1pf(0x1.7ffffep+127));

    //f64x4: Log max_ulp 1.99786e+17
    //worst_input=3.478743 (0x400bd47719a77412), worst_output=-43.114749 (0xc0458eb014fa1256)
    //double tough_input = 0x1.bd47719a77412p+1;
    double tough_input = 340282346638528859811704183484516925440.000000;
    // namespace hn = ::hwy::N_AVX2;
    // hn::ScalableTag<double> d;
    // printf("Hwy result SSE4: %a\n", hn::GetLane(hn::sleef::Log(d, hn::Set(d, tough_input))));

    hwy::N_SSE4::ScalableTag<float> davx;
    printf("Hwy result AVX2: %a\n", hwy::N_SSE4::GetLane(hwy::N_SSE4::sleef::Log1p(davx, hwy::N_SSE4::Set(davx, tough_input))));
    // for (int i = 0; i < hwy::N_AVX2::Lanes(davx); i++) {
    //     printf("Hwy result AVX2[%i]: %a\n", i, hwy::N_AVX2::ExtractLane(hwy::N_AVX2::sleef::Log(davx, hwy::N_AVX2::Set(davx, tough_input)), i));
    // }

    printf("Sleef scalar result: %a\n", Sleef_log1pf1_u10(tough_input));
    printf("Sleef vector sse4: %a\n", _mm_cvtss_f32(Sleef_log1pf4_u10sse4(_mm_set1_ps(tough_input))));
    printf("Correct result: %a\n", std::log1p(tough_input));
}
#endif
