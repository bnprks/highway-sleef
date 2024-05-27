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

//#define PI 3.1415926535897932384626433832795028841971693993751L
#define PI 3.141592653589793238462643383279502884L
// Copy the approach of boost for now
template <typename T>
static T sinpi(T x) {
    if (x < 0) return -sinpi(-x);
    if (x < 0.5) return std::sin(PI * x);
    bool invert = x < 1;
    if (x < 1) x = -x;

    T rem = std::floor(x);
    // Invert odd remainders
    if (std::abs(std::floor(rem/2)*2 - rem) > std::numeric_limits<T>::epsilon()) {
        invert = !invert;
    }
    rem = x - rem;
    if (rem > 0.5) rem = 1-rem;
    if (rem == 0.5) return invert ? -1 : 1;
    T result = std::sin(PI * rem);
    return invert ? -result : result;
}
template <typename T>
static T cospi(T x) {
    bool invert = false;
    if (std::abs(x) < 0.25) return std::cos(PI * x);
    x = std::abs(x);
    T rem = std::floor(x);
    // Invert odd remainders
    if (std::abs(std::floor(rem/2)*2 - rem) > std::numeric_limits<T>::epsilon()) {
        invert = !invert;
    }
    rem = x - rem;
    if (rem > 0.5) {
        rem = 1- rem;
        invert = !invert;
    }
    if (rem == 0.5) {
        return 0;
    }
    if (rem > 0.25) {
        rem = 0.5 - rem;
        rem = std::sin(PI * rem);
    } else {
        rem = std::cos(PI * rem);
    }
    return invert ? -rem : rem;
}
#undef PI

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

    // f64x4: SinPi max_ulp 1.09766
    //     worst_input=0.000000 (0xeee8dd026da3f), worst_output=0.000000 (0x27747426aebefc)
    // worst output: 0x1.7747426aebefcp-1021
    // double i1 = -0x1.d95c0ep-126;
    // double i2 = 0x1.9ed4d6p-125;
    double tough_input = -1.0;
    // namespace hn = ::hwy::N_AVX2;
    // hn::ScalableTag<double> d;
    // printf("Hwy result SSE4: %a\n", hn::GetLane(hn::sleef::Log(d, hn::Set(d, tough_input))));

    hwy::N_AVX2::ScalableTag<double> davx2;
    // printf("Hwy result AVX2: %a\n", hwy::N_AVX2::GetLane(hwy::N_AVX2::sleef::SinPi(davx2, hwy::N_AVX2::Set(davx2, tough_input))));
    // printf("Hwy result AVX2: %a\n", hwy::N_AVX2::GetLane(hwy::N_AVX2::sleef::Hypot(davx2, hwy::N_AVX2::Set(davx2, i1), hwy::N_AVX2::Set(davx2, i2))));
    // printf("Hwy result AVX2: %.20g\n", hwy::N_AVX2::GetLane(hwy::N_AVX2::sleef::Hypot(davx2, hwy::N_AVX2::Set(davx2, i1), hwy::N_AVX2::Set(davx2, i2))));

    hwy::N_SSE4::ScalableTag<double> dsse4;
    printf("Hwy result SSE4: %a\n", hwy::N_SSE4::GetLane(hwy::N_SSE4::sleef::SinPi(dsse4, hwy::N_SSE4::Set(dsse4, tough_input))));
    // printf("Hwy result SSE4: %a\n", hwy::N_SSE4::GetLane(hwy::N_SSE4::sleef::Hypot(dsse4, hwy::N_SSE4::Set(dsse4, i1), hwy::N_SSE4::Set(dsse4, i2))));
    // printf("Hwy result SSE4: %.20g\n", hwy::N_SSE4::GetLane(hwy::N_SSE4::sleef::Hypot(dsse4, hwy::N_SSE4::Set(dsse4, i1), hwy::N_SSE4::Set(dsse4, i2))));


    // printf("Sleef scalar result: %.20g\n", Sleef_sinpi_u05(tough_input));
    // printf("Sleef vector avx2: %.20g\n", _mm256_cvtsd_f64(Sleef_sinpid4_u05avx2(_mm256_set1_pd(tough_input))));
    printf("Sleef vector sse4: %.20g\n", _mm_cvtsd_f64(Sleef_sinpid2_u05sse4(_mm_set1_pd(tough_input))));
    printf("Correct result: %.20Lg\n", sinpi((long double) tough_input));
    printf("Correct result (double): %.20g\n", (double) sinpi((long double) tough_input));
    printf("Correct result (double): %a\n", (double) sinpi((long double) tough_input));

    //printf("Sleef scalar result: %.20g\n", Sleef_powd1_u10(i1, i2));
    // printf("Sleef vector avx2: %.20g\n", _mm256_cvtsd_f64(Sleef_powd4_u10avx2(_mm256_set1_pd(i1), _mm256_set1_pd(i2))));
    // printf("Sleef vector sse4: %.20g\n", _mm_cvtsd_f64(Sleef_powd2_u10sse4(_mm_set1_pd(i1), _mm_set1_pd(i2))));
    // printf("Sleef vector avx2: %.20g\n", _mm256_cvtss_f32(Sleef_powf8_u10avx2(_mm256_set1_ps(i1), _mm256_set1_ps(i2))));
    // printf("Sleef vector avx2: %a\n", _mm256_cvtss_f32(Sleef_hypotf8_u05avx2(_mm256_set1_ps(i1), _mm256_set1_ps(i2))));
    // printf("Sleef vector sse4: %.20g\n", _mm_cvtss_f32(Sleef_hypotf4_u05sse4(_mm_set1_ps(i1), _mm_set1_ps(i2))));
    //printf("Sleef vector sse4: %.20g\n", _mm_cvtss_f32(Sleef_tgammaf4_u10sse4(_mm_set1_ps(tough_input))));
    // printf("Correct result: %.20Lg\n", std::hypot((long double) i1, (long double) i2));
    // printf("Correct result (double): %.20g\n", (double) std::hypot((long double) i1, (long double) i2));
    // printf("Correct result (double): %a\n", (double) std::hypot((long double) i1, (long double) i2));
}
#endif
