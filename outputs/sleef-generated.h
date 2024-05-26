
// This file is translated from the SLEEF vectorized math library.
// Translation performed by Ben Parks copyright 2024.
// Translated elements available under the following licenses, at your option:
//   BSL-1.0 (http://www.boost.org/LICENSE_1_0.txt),
//   MIT (https://opensource.org/license/MIT/), and
//   Apache-2.0 (https://www.apache.org/licenses/LICENSE-2.0)
// 
// Original SLEEF copyright:
//   Copyright Naoki Shibata and contributors 2010 - 2021.
// Distributed under the Boost Software License, Version 1.0.
//    (See accompanying file LICENSE.txt or copy at
//          http://www.boost.org/LICENSE_1_0.txt)

#if defined(HIGHWAY_HWY_CONTRIB_SLEEF_SLEEF_INL_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_SLEEF_SLEEF_INL_
#undef HIGHWAY_HWY_CONTRIB_SLEEF_SLEEF_INL_
#else
#define HIGHWAY_HWY_CONTRIB_SLEEF_SLEEF_INL_
#endif

#include <type_traits>
#include "hwy/highway.h"

extern const float PayneHanekReductionTable_float[]; // Precomputed table of exponent values for Payne Hanek reduction
extern const double PayneHanekReductionTable_double[]; // Precomputed table of exponent values for Payne Hanek reduction

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
HWY_API Vec512<float> GetExponent(Vec512<float> x) {
  return Vec512<float>{_mm512_getexp_ps(x.raw)};
}
HWY_API Vec256<float> GetExponent(Vec256<float> x) {
  return Vec256<float>{_mm256_getexp_ps(x.raw)};
}
template<size_t N>
HWY_API Vec128<float, N> GetExponent(Vec128<float, N> x) {
  return Vec128<float, N>{_mm_getexp_ps(x.raw)};
}

HWY_API Vec512<double> GetExponent(Vec512<double> x) {
  return Vec512<double>{_mm512_getexp_pd(x.raw)};
}
HWY_API Vec256<double> GetExponent(Vec256<double> x) {
  return Vec256<double>{_mm256_getexp_pd(x.raw)};
}
template<size_t N>
HWY_API Vec128<double, N> GetExponent(Vec128<double, N> x) {
  return Vec128<double, N>{_mm_getexp_pd(x.raw)};
}

HWY_API Vec512<float> GetMantissa(Vec512<float> x) {
  return Vec512<float>{_mm512_getmant_ps(x.raw,  _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)};
}
HWY_API Vec256<float> GetMantissa(Vec256<float> x) {
  return Vec256<float>{_mm256_getmant_ps(x.raw,  _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)};
}
template<size_t N>
HWY_API Vec128<float, N> GetMantissa(Vec128<float, N> x) {
  return Vec128<float, N>{_mm_getmant_ps(x.raw,  _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)};
}

HWY_API Vec512<double> GetMantissa(Vec512<double> x) {
  return Vec512<double>{_mm512_getmant_pd(x.raw,  _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)};
}
HWY_API Vec256<double> GetMantissa(Vec256<double> x) {
  return Vec256<double>{_mm256_getmant_pd(x.raw,  _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)};
}
template<size_t N>
HWY_API Vec128<double, N> GetMantissa(Vec128<double, N> x) {
  return Vec128<double, N>{_mm_getmant_pd(x.raw,  _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)};
}

template<int I>
HWY_API Vec512<float> Fixup(Vec512<float> a, Vec512<float> b, Vec512<int> c) {
    return Vec512<float>{_mm512_fixupimm_ps(a.raw, b.raw, c.raw, I)};
}
template<int I>
HWY_API Vec256<float> Fixup(Vec256<float> a, Vec256<float> b, Vec256<int> c) {
    return Vec256<float>{_mm256_fixupimm_ps(a.raw, b.raw, c.raw, I)};
}
template<int I, size_t N>
HWY_API Vec128<float, N> Fixup(Vec128<float, N> a, Vec128<float, N> b, Vec128<int, N> c) {
    return Vec128<float, N>{_mm_fixupimm_ps(a.raw, b.raw, c.raw, I)};
}

template<int I>
HWY_API Vec512<double> Fixup(Vec512<double> a, Vec512<double> b, Vec512<int64_t> c) {
    return Vec512<double>{_mm512_fixupimm_pd(a.raw, b.raw, c.raw, I)};
}
template<int I>
HWY_API Vec256<double> Fixup(Vec256<double> a, Vec256<double> b, Vec256<int64_t> c) {
    return Vec256<double>{_mm256_fixupimm_pd(a.raw, b.raw, c.raw, I)};
}
template<int I, size_t N>
HWY_API Vec128<double, N> Fixup(Vec128<double, N> a, Vec128<double, N> b, Vec128<int64_t, N> c) {
    return Vec128<double, N>{_mm_fixupimm_pd(a.raw, b.raw, c.raw, I)};
}
#endif

namespace sleef {

#undef HWY_SLEEF_HAS_FMA
#if (HWY_ARCH_X86 && HWY_TARGET < HWY_SSE4) || HWY_ARCH_ARM || HWY_ARCH_S390X || HWY_ARCH_RVV 
#define HWY_SLEEF_HAS_FMA 1
#endif

#undef HWY_SLEEF_IF_DOUBLE
#define HWY_SLEEF_IF_DOUBLE(D, V) typename std::enable_if<std::is_same<double, TFromD<D>>::value, V>::type
#undef HWY_SLEEF_IF_FLOAT
#define HWY_SLEEF_IF_FLOAT(D, V) typename std::enable_if<std::is_same<float, TFromD<D>>::value, V>::type

// Computes e^x
// Translated from libm/sleefsimdsp.c:2023 xexpf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Exp(const D df, Vec<D> d);

// Computes 2^x
// Translated from libm/sleefsimdsp.c:2605 xexp2f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Exp2(const D df, Vec<D> d);

// Computes 10^x
// Translated from libm/sleefsimdsp.c:2654 xexp10f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Exp10(const D df, Vec<D> d);

// Computes e^x - 1
// Translated from libm/sleefsimdsp.c:2701 xexpm1f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Expm1(const D df, Vec<D> a);

// Computes ln(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2268 xlogf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Log(const D df, Vec<D> d);

// Computes ln(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1984 xlogf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) LogFast(const D df, Vec<D> d);

// Computes log10(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2712 xlog10f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Log10(const D df, Vec<D> d);

// Computes log2(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2757 xlog2f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Log2(const D df, Vec<D> d);

// Computes log1p(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2842 xlog1pf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Log1p(const D df, Vec<D> d);

// Computes sqrt(x) with 0.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:2992 xsqrtf_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Sqrt(const D df, Vec<D> d);

// Computes sqrt(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:2073 xsqrtf_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) SqrtFast(const D df, Vec<D> d);

// Computes cube root of x with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:2100 xcbrtf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) CbrtFast(const D df, Vec<D> d);

// Computes cube root of x with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2141 xcbrtf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Cbrt(const D df, Vec<D> d);

// Computes sqrt(x^2 + y^2) with 0.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:3069 xhypotf_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Hypot(const D df, Vec<D> x, Vec<D> y);

// Computes sqrt(x^2 + y^2) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:3090 xhypotf_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) HypotFast(const D df, Vec<D> x, Vec<D> y);

// Computes x^y with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2360 xpowf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Pow(const D df, Vec<D> x, Vec<D> y);

// Computes sin(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:969 xsinf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Sin(const D df, Vec<D> d);

// Computes cos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1067 xcosf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Cos(const D df, Vec<D> d);

// Computes tan(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1635 xtanf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Tan(const D df, Vec<D> d);

// Computes sin(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:630 xsinf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) SinFast(const D df, Vec<D> d);

// Computes cos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:736 xcosf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) CosFast(const D df, Vec<D> d);

// Computes tan(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:845 xtanf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) TanFast(const D df, Vec<D> d);

// Computes sinh(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2447 xsinhf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Sinh(const D df, Vec<D> x);

// Computes cosh(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2461 xcoshf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Cosh(const D df, Vec<D> x);

// Computes tanh(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2474 xtanhf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Tanh(const D df, Vec<D> x);

// Computes sinh(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:2489 xsinhf_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) SinhFast(const D df, Vec<D> x);

// Computes cosh(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:2502 xcoshf_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) CoshFast(const D df, Vec<D> x);

// Computes tanh(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:2513 xtanhf_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) TanhFast(const D df, Vec<D> x);

// Computes acos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1948 xacosf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Acos(const D df, Vec<D> d);

// Computes asin(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1928 xasinf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Asin(const D df, Vec<D> d);

// Computes asinh(x) with 1 ULP accuracy
// Translated from libm/sleefsimdsp.c:2554 xasinhf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Asinh(const D df, Vec<D> x);

// Computes acos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1847 xacosf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) AcosFast(const D df, Vec<D> d);

// Computes asin(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1831 xasinf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) AsinFast(const D df, Vec<D> d);

// Computes atan(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1743 xatanf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) AtanFast(const D df, Vec<D> d);

// Computes acosh(x) with 1 ULP accuracy
// Translated from libm/sleefsimdsp.c:2575 xacoshf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Acosh(const D df, Vec<D> x);

// Computes atan(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1973 xatanf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Atan(const D df, Vec<D> d);

// Computes atanh(x) with 1 ULP accuracy
// Translated from libm/sleefsimdsp.c:2591 xatanhf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Atanh(const D df, Vec<D> x);

// Computes e^x
// Translated from libm/sleefsimddp.c:2146 xexp
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Exp(const D df, Vec<D> d);

// Computes 2^x
// Translated from libm/sleefsimddp.c:2686 xexp2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Exp2(const D df, Vec<D> d);

// Computes 10^x
// Translated from libm/sleefsimddp.c:2750 xexp10
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Exp10(const D df, Vec<D> d);

// Computes e^x - 1
// Translated from libm/sleefsimddp.c:2815 xexpm1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Expm1(const D df, Vec<D> a);

// Computes ln(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2270 xlog_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Log(const D df, Vec<D> d);

// Computes ln(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:2099 xlog
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) LogFast(const D df, Vec<D> d);

// Computes log10(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2824 xlog10
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Log10(const D df, Vec<D> d);

// Computes log2(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2875 xlog2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Log2(const D df, Vec<D> d);

// Computes log1p(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2974 xlog1p
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Log1p(const D df, Vec<D> d);

// Computes sqrt(x) with 0.5 ULP accuracy
// Translated from libm/sleefsimddp.c:3142 xsqrt_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Sqrt(const D df, Vec<D> d);

// Computes sqrt(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:3220 xsqrt_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) SqrtFast(const D df, Vec<D> d);

// Computes cube root of x with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:2588 xcbrt
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) CbrtFast(const D df, Vec<D> d);

// Computes cube root of x with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2630 xcbrt_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Cbrt(const D df, Vec<D> d);

// Computes sqrt(x^2 + y^2) with 0.5 ULP accuracy
// Translated from libm/sleefsimddp.c:3222 xhypot_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Hypot(const D df, Vec<D> x, Vec<D> y);

// Computes sqrt(x^2 + y^2) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:3243 xhypot_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) HypotFast(const D df, Vec<D> x, Vec<D> y);

// Computes x^y with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2358 xpow
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Pow(const D df, Vec<D> x, Vec<D> y);

// Computes sin(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:521 xsin_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Sin(const D df, Vec<D> d);

// Computes cos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:787 xcos_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Cos(const D df, Vec<D> d);

// Computes tan(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:1645 xtan_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Tan(const D df, Vec<D> d);

// Computes sin(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:382 xsin
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) SinFast(const D df, Vec<D> d);

// Computes cos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:652 xcos
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) CosFast(const D df, Vec<D> d);

// Computes tan(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:1517 xtan
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) TanFast(const D df, Vec<D> d);

// Computes sinh(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2435 xsinh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Sinh(const D df, Vec<D> x);

// Computes cosh(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2448 xcosh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Cosh(const D df, Vec<D> x);

// Computes tanh(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2460 xtanh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Tanh(const D df, Vec<D> x);

// Computes sinh(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:2474 xsinh_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) SinhFast(const D df, Vec<D> x);

// Computes cosh(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:2487 xcosh_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) CoshFast(const D df, Vec<D> x);

// Computes tanh(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:2497 xtanh_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) TanhFast(const D df, Vec<D> x);

// Computes acos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2007 xacos_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Acos(const D df, Vec<D> d);

// Computes asin(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:1945 xasin_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Asin(const D df, Vec<D> d);

// Computes asinh(x) with 1 ULP accuracy
// Translated from libm/sleefsimddp.c:2539 xasinh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Asinh(const D df, Vec<D> x);

// Computes acos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:1975 xacos
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) AcosFast(const D df, Vec<D> d);

// Computes asin(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:1919 xasin
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) AsinFast(const D df, Vec<D> d);

// Computes atan(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:2049 xatan
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) AtanFast(const D df, Vec<D> s);

// Computes acosh(x) with 1 ULP accuracy
// Translated from libm/sleefsimddp.c:2561 xacosh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Acosh(const D df, Vec<D> x);

// Computes atan(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2042 xatan_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Atan(const D df, Vec<D> d);

// Computes atanh(x) with 1 ULP accuracy
// Translated from libm/sleefsimddp.c:2576 xatanh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Atanh(const D df, Vec<D> x);

namespace {

template<class D>
using RebindToSigned32 = Rebind<int32_t, D>;
template<class D>
using RebindToUnsigned32 = Rebind<uint32_t, D>;

// Estrin's Scheme is a faster method for evaluating large polynomials on
// super scalar architectures. It works by factoring the Horner's Method
// polynomial into power of two sub-trees that can be evaluated in parallel.
// Wikipedia Link: https://en.wikipedia.org/wiki/Estrin%27s_scheme
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T c0, T c1) {
  return MulAdd(c1, x, c0);
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T c0, T c1, T c2) {
  return MulAdd(x2, c2, MulAdd(c1, x, c0));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T c0, T c1, T c2, T c3) {
  return MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T c0, T c1, T c2, T c3, T c4) {
  return MulAdd(x4, c4, MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T c0, T c1, T c2, T c3, T c4, T c5) {
  return MulAdd(x4, MulAdd(c5, x, c4),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6) {
  return MulAdd(x4, MulAdd(x2, c6, MulAdd(c5, x, c4)),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7) {
  return MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8) {
  return MulAdd(x8, c8,
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9) {
  return MulAdd(x8, MulAdd(c9, x, c8),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10) {
  return MulAdd(x8, MulAdd(x2, c10, MulAdd(c9, x, c8)),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11) {
  return MulAdd(x8, MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8)),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12) {
  return MulAdd(
      x8, MulAdd(x4, c12, MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
      MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
             MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13) {
  return MulAdd(x8,
                MulAdd(x4, MulAdd(c13, x, c12),
                       MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14) {
  return MulAdd(x8,
                MulAdd(x4, MulAdd(x2, c14, MulAdd(c13, x, c12)),
                       MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14, T c15) {
  return MulAdd(x8,
                MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                       MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T x16, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14, T c15, T c16) {
  return MulAdd(
      x16, c16,
      MulAdd(x8,
             MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                    MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
             MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                    MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T x16, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14, T c15, T c16, T c17) {
  return MulAdd(
      x16, MulAdd(c17, x, c16),
      MulAdd(x8,
             MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                    MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
             MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                    MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T x16, T c0, T c1, T c2, T c3, T c4, T c5,
                                     T c6, T c7, T c8, T c9, T c10, T c11,
                                     T c12, T c13, T c14, T c15, T c16, T c17,
                                     T c18) {
  return MulAdd(
      x16, MulAdd(x2, c18, MulAdd(c17, x, c16)),
      MulAdd(x8,
             MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                    MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
             MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                    MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)))));
}

//////////////////
// Constants
//////////////////
constexpr double Pi = 3.141592653589793238462643383279502884; // pi
constexpr float OneOverPi = 0.318309886183790671537767526745028724; // 1 / pi
constexpr float FloatMin = 0x1p-126; // Minimum normal float value
constexpr double DoubleMin = 0x1p-1022; // Minimum normal double value
constexpr double PiA = 3.1415926218032836914; // Four-part sum of Pi (1/4)
constexpr double PiB = 3.1786509424591713469e-08; // Four-part sum of Pi (2/4)
constexpr double PiC = 1.2246467864107188502e-16; // Four-part sum of Pi (3/4)
constexpr double PiD = 1.2736634327021899816e-24; // Four-part sum of Pi (4/4)
constexpr double TrigRangeMax = 1e+14; // Max value for using 4-part sum of Pi
constexpr double PiA2 = 3.141592653589793116; // Three-part sum of Pi (1/3)
constexpr double PiB2 = 1.2246467991473532072e-16; // Three-part sum of Pi (2/3)
constexpr double TrigRangeMax2 = 15; // Max value for using 3-part sum of Pi
constexpr double TwoOverPiH = 0.63661977236758138243; // Two-part sum of 2 / pi (1/2)
constexpr double TwoOverPiL = -3.9357353350364971764e-17; // Two-part sum of 2 / pi (2/2)
constexpr double SqrtDoubleMax = 1.3407807929942596355e+154; // Square root of max DP floating-point number
constexpr double Ln2Hi_d = .69314718055966295651160180568695068359375; // Ln2Hi + Ln2Lo ~= ln(2)
constexpr double Ln2Lo_d = .28235290563031577122588448175013436025525412068e-12; // Ln2Hi + Ln2Lo ~= ln(2)
constexpr double OneOverLn2_d = 1.442695040888963407359924681001892137426645954152985934135449406931; // 1 / ln(2)
constexpr double Log10Of2_Hi_d = 0.30102999566383914498 ; // Log10Of2_Hi_d + Log10Of2_Lo_d ~= log10(2)
constexpr double Log10Of2_Lo_d = 1.4205023227266099418e-13; // Log10Of2_Hi_d + Log10Of2_Lo_d ~= log10(2)
constexpr double Lg10 = 3.3219280948873623478703194294893901758648313930; // log2(10)
constexpr float Log10Of2_Hi_f = 0.3010253906f; // Log10Of2_Hi_f + Log10Of2_Lo_f ~= log10(2)
constexpr float Log10Of2_Lo_f = 4.605038981e-06f; // Log10Of2_Hi_f + Log10Of2_Lo_f ~= log10(2)
constexpr float PiAf = 3.140625f; // Four-part sum of Pi (1/4)
constexpr float PiBf = 0.0009670257568359375f; // Four-part sum of Pi (2/4)
constexpr float PiCf = 6.2771141529083251953e-07f; // Four-part sum of Pi (3/4)
constexpr float PiDf = 1.2154201256553420762e-10f; // Four-part sum of Pi (4/4)
constexpr float TrigRangeMaxf = 39000; // Max value for using 4-part sum of Pi
constexpr float PiA2f = 3.1414794921875f; // Three-part sum of Pi (1/3)
constexpr float PiB2f = 0.00011315941810607910156f; // Three-part sum of Pi (2/3)
constexpr float PiC2f = 1.9841872589410058936e-09f; // Three-part sum of Pi (3/3)
constexpr float TrigRangeMax2f = 125.0f; // Max value for using 3-part sum of Pi
constexpr float SqrtFloatMax = 18446743523953729536.0; // Square root of max floating-point number
constexpr float Ln2Hi_f = 0.693145751953125f; // Ln2Hi + Ln2Lo ~= ln(2)
constexpr float Ln2Lo_f = 1.428606765330187045e-06f; // Ln2Hi + Ln2Lo ~= ln(2)
constexpr float OneOverLn2_f = 1.442695040888963407359924681001892137426645954152985934135449406931f; // 1 / ln(2)
constexpr float Pif = ((float)M_PI); // pi (float)
#if (defined (__GNUC__) || defined (__clang__) || defined(__INTEL_COMPILER)) && !defined(_MSC_VER)
constexpr double NanDouble = __builtin_nan(""); // Double precision NaN
constexpr float NanFloat = __builtin_nanf(""); // Floating point NaN
constexpr double InfDouble = __builtin_inf(); // Double precision infinity
constexpr float InfFloat = __builtin_inff(); // Floating point infinity
#elif defined(_MSC_VER) 
constexpr double InfDouble = (1e+300 * 1e+300); // Double precision infinity
constexpr double NanDouble = (SLEEF_INFINITY - SLEEF_INFINITY); // Double precision NaN
constexpr float InfFloat = ((float)SLEEF_INFINITY); // Floating point infinity
constexpr float NanFloat = ((float)SLEEF_NAN); // Floating point NaN
#endif


// Computes 2^x, where x is an integer.
// Translated from libm/sleefsimdsp.c:516 vpow2i_vf_vi2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Pow2I(const D df, Vec<RebindToSigned<D>> q) {
  RebindToSigned<D> di;
  
  return BitCast(df, ShiftLeft<23>(Add(q, Set(di, 0x7f))));
}

// Sets the exponent of 'x' to 2^e. Fast, but "short reach"
// Translated from libm/sleefsimdsp.c:535 vldexp2_vf_vf_vi2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) LoadExp2(const D df, Vec<D> d, Vec<RebindToSigned<D>> e) {
  return Mul(Mul(d, Pow2I(df, ShiftRight<1>(e))), Pow2I(df, Sub(e, ShiftRight<1>(e))));
}

// Add (v0 + 1) + v2
// Translated from common/df.h:59 vadd_vf_3vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Add3(const D df, Vec<D> v0, Vec<D> v1, Vec<D> v2) {
  return Add(Add(v0, v1), v2);
}

// Computes x + y in double-float precision, sped up by assuming |x| > |y|
// Translated from common/df.h:146 dfadd_vf2_vf_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) AddFastDF(const D df, Vec<D> x, Vec2<D> y) {
  Vec<D> s = Add(x, Get2<0>(y));
  return Create2(df, s, Add3(df, Sub(x, s), Get2<0>(y), Get2<1>(y)));
}

// Normalizes a double-float precision representation (redistributes hi vs. lo value)
// Translated from common/df.h:102 dfnormalize_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) NormalizeDF(const D df, Vec2<D> t) {
  Vec<D> s = Add(Get2<0>(t), Get2<1>(t));
  return Create2(df, s, Add(Sub(Get2<0>(t), s), Get2<1>(t)));
}

// Set the bottom half of mantissa bits to 0 (used in some double-float math)
// Translated from common/df.h:22 vupper_vf_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) LowerPrecision(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  return BitCast(df, And(BitCast(di, d), Set(di, 0xfffff000)));
}

// Computes x * y in double-float precision
// Translated from common/df.h:191 dfmul_vf2_vf_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) MulDF(const D df, Vec<D> x, Vec<D> y) {
#if HWY_SLEEF_HAS_FMA
  Vec<D> s = Mul(x, y);
  return Create2(df, s, MulSub(x, y, s));
#else
  Vec<D> xh = LowerPrecision(df, x), xl = Sub(x, xh);
  Vec<D> yh = LowerPrecision(df, y), yl = Sub(y, yh);

  Vec<D> s = Mul(x, y), t;

  t = MulAdd(xh, yh, Neg(s));
  t = MulAdd(xl, yh, t);
  t = MulAdd(xh, yl, t);
  t = MulAdd(xl, yl, t);

  return Create2(df, s, t);
#endif
}

// Computes x + y in double-float precision, sped up by assuming |x| > |y|
// Translated from common/df.h:129 dfadd_vf2_vf2_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) AddFastDF(const D df, Vec2<D> x, Vec<D> y) {
  Vec<D> s = Add(Get2<0>(x), y);
  return Create2(df, s, Add3(df, Sub(Get2<0>(x), s), y, Get2<1>(x)));
}

// Computes x * y in double-float precision
// Translated from common/df.h:214 dfmul_vf2_vf2_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) MulDF(const D df, Vec2<D> x, Vec<D> y) {
#if HWY_SLEEF_HAS_FMA
  Vec<D> s = Mul(Get2<0>(x), y);
  return Create2(df, s, MulAdd(Get2<1>(x), y, MulSub(Get2<0>(x), y, s)));
#else
  Vec<D> xh = LowerPrecision(df, Get2<0>(x)), xl = Sub(Get2<0>(x), xh);
  Vec<D> yh = LowerPrecision(df, y  ), yl = Sub(y, yh);

  Vec<D> s = Mul(Get2<0>(x), y), t;

  t = MulAdd(xh, yh, Neg(s));
  t = MulAdd(xl, yh, t);
  t = MulAdd(xh, yl, t);
  t = MulAdd(xl, yl, t);
  t = MulAdd(Get2<1>(x), y, t);

  return Create2(df, s, t);
#endif
}

// Computes x + y in double-float precision
// Translated from common/df.h:139 dfadd2_vf2_vf2_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) AddDF(const D df, Vec2<D> x, Vec<D> y) {
  Vec<D> s = Add(Get2<0>(x), y);
  Vec<D> v = Sub(s, Get2<0>(x));
  Vec<D> t = Add(Sub(Get2<0>(x), Sub(s, v)), Sub(y, v));
  return Create2(df, s, Add(t, Get2<1>(x)));
}

// Computes x^2 in double-float precision
// Translated from common/df.h:196 dfsqu_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SquareDF(const D df, Vec2<D> x) {
#if HWY_SLEEF_HAS_FMA
  Vec<D> s = Mul(Get2<0>(x), Get2<0>(x));
  return Create2(df, s, MulAdd(Add(Get2<0>(x), Get2<0>(x)), Get2<1>(x), MulSub(Get2<0>(x), Get2<0>(x), s)));
#else
  Vec<D> xh = LowerPrecision(df, Get2<0>(x)), xl = Sub(Get2<0>(x), xh);

  Vec<D> s = Mul(Get2<0>(x), Get2<0>(x)), t;

  t = MulAdd(xh, xh, Neg(s));
  t = MulAdd(Add(xh, xh), xl, t);
  t = MulAdd(xl, xl, t);
  t = MulAdd(Get2<0>(x), Add(Get2<1>(x), Get2<1>(x)), t);

  return Create2(df, s, t);
#endif
}

// Computes x * y in double-float precision
// Translated from common/df.h:205 dfmul_vf2_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) MulDF(const D df, Vec2<D> x, Vec2<D> y) {
#if HWY_SLEEF_HAS_FMA
  Vec<D> s = Mul(Get2<0>(x), Get2<0>(y));
  return Create2(df, s, MulAdd(Get2<0>(x), Get2<1>(y), MulAdd(Get2<1>(x), Get2<0>(y), MulSub(Get2<0>(x), Get2<0>(y), s))));
#else
  Vec<D> xh = LowerPrecision(df, Get2<0>(x)), xl = Sub(Get2<0>(x), xh);
  Vec<D> yh = LowerPrecision(df, Get2<0>(y)), yl = Sub(Get2<0>(y), yh);

  Vec<D> s = Mul(Get2<0>(x), Get2<0>(y)), t;

  t = MulAdd(xh, yh, Neg(s));
  t = MulAdd(xl, yh, t);
  t = MulAdd(xh, yl, t);
  t = MulAdd(xl, yl, t);
  t = MulAdd(Get2<0>(x), Get2<1>(y), t);
  t = MulAdd(Get2<1>(x), Get2<0>(y), t);

  return Create2(df, s, t);
#endif
}

// Computes x + y in double-float precision
// Translated from common/df.h:158 dfadd2_vf2_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) AddDF(const D df, Vec2<D> x, Vec2<D> y) {
  Vec<D> s = Add(Get2<0>(x), Get2<0>(y));
  Vec<D> v = Sub(s, Get2<0>(x));
  Vec<D> t = Add(Sub(Get2<0>(x), Sub(s, v)), Sub(Get2<0>(y), v));
  return Create2(df, s, Add(t, Add(Get2<1>(x), Get2<1>(y))));
}

// Computes e^x in double-float precision
// Translated from libm/sleefsimdsp.c:2418 expk2f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) ExpDF(const D df, Vec2<D> d) {
  RebindToUnsigned<D> du;
  
  Vec<D> u = Mul(Add(Get2<0>(d), Get2<1>(d)), Set(df, OneOverLn2_f));
  Vec<RebindToSigned<D>> q = NearestInt(u);
  Vec2<D> s, t;

  s = AddDF(df, d, Mul(ConvertTo(df, q), Set(df, -Ln2Hi_f)));
  s = AddDF(df, s, Mul(ConvertTo(df, q), Set(df, -Ln2Lo_f)));

  u = Set(df, +0.1980960224e-3f);
  u = MulAdd(u, Get2<0>(s), Set(df, +0.1394256484e-2f));
  u = MulAdd(u, Get2<0>(s), Set(df, +0.8333456703e-2f));
  u = MulAdd(u, Get2<0>(s), Set(df, +0.4166637361e-1f));

  t = AddDF(df, MulDF(df, s, u), Set(df, +0.166666659414234244790680580464e+0f));
  t = AddDF(df, MulDF(df, s, t), Set(df, 0.5));
  t = AddDF(df, s, MulDF(df, SquareDF(df, s), t));

  t = AddFastDF(df, Set(df, 1), t);

  t = Set2<0>(t, LoadExp2(df, Get2<0>(t), q));
  t = Set2<1>(t, LoadExp2(df, Get2<1>(t), q));

  t = Set2<0>(t, BitCast(df, IfThenZeroElse(RebindMask(du, Lt(Get2<0>(d), Set(df, -104))), BitCast(du, Get2<0>(t)))));
  t = Set2<1>(t, BitCast(df, IfThenZeroElse(RebindMask(du, Lt(Get2<0>(d), Set(df, -104))), BitCast(du, Get2<1>(t)))));

  return t;
}

// Computes x / y in double-float precision
// Translated from common/df.h:183 dfdiv_vf2_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) DivDF(const D df, Vec2<D> n, Vec2<D> d) {
#if HWY_SLEEF_HAS_FMA
  Vec<D> t = Div(Set(df, 1.0), Get2<0>(d));
  Vec<D> s = Mul(Get2<0>(n), t);
  Vec<D> u = MulSub(t, Get2<0>(n), s);
  Vec<D> v = NegMulAdd(Get2<1>(d), t, NegMulAdd(Get2<0>(d), t, Set(df, 1)));
  return Create2(df, s, MulAdd(s, v, MulAdd(Get2<1>(n), t, u)));
#else
  Vec<D> t = Div(Set(df, 1.0), Get2<0>(d));
  Vec<D> dh  = LowerPrecision(df, Get2<0>(d)), dl  = Sub(Get2<0>(d), dh);
  Vec<D> th  = LowerPrecision(df, t  ), tl  = Sub(t, th);
  Vec<D> nhh = LowerPrecision(df, Get2<0>(n)), nhl = Sub(Get2<0>(n), nhh);

  Vec<D> s = Mul(Get2<0>(n), t);

  Vec<D> u, w;
  w = Set(df, -1);
  w = MulAdd(dh, th, w);
  w = MulAdd(dh, tl, w);
  w = MulAdd(dl, th, w);
  w = MulAdd(dl, tl, w);
  w = Neg(w);

  u = MulAdd(nhh, th, Neg(s));
  u = MulAdd(nhh, tl, u);
  u = MulAdd(nhl, th, u);
  u = MulAdd(nhl, tl, u);
  u = MulAdd(s, w, u);

  return Create2(df, s, MulAdd(t, Sub(Get2<1>(n), Mul(s, Get2<1>(d))), u));
#endif
}

// Sets the exponent of 'x' to 2^e. Very fast, "no denormal"
// Translated from libm/sleefsimdsp.c:539 vldexp3_vf_vf_vi2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) LoadExp3(const D df, Vec<D> d, Vec<RebindToSigned<D>> q) {
  RebindToSigned<D> di;
  
  return BitCast(df, Add(BitCast(di, d), ShiftLeft<23>(q)));
}

// Integer log of x, "but the argument must be a normalized value"
// Translated from libm/sleefsimdsp.c:497 vilogb2k_vi2_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<RebindToSigned<D>>) ILogB2(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<RebindToSigned<D>> q = BitCast(di, d);
  q = BitCast(di, ShiftRight<23>(BitCast(du, q)));
  q = And(q, Set(di, 0xff));
  q = Sub(q, Set(di, 0x7f));
  return q;
}

// Add ((v0 + 1) + v2) + v3
// Translated from common/df.h:63 vadd_vf_4vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Add4(const D df, Vec<D> v0, Vec<D> v1, Vec<D> v2, Vec<D> v3) {
  return Add3(df, Add(v0, v1), v2, v3);
}

// Computes x + y in double-float precision, sped up by assuming |x| > |y|
// Translated from common/df.h:151 dfadd_vf2_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) AddFastDF(const D df, Vec2<D> x, Vec2<D> y) {
  // |x| >= |y|

  Vec<D> s = Add(Get2<0>(x), Get2<0>(y));
  return Create2(df, s, Add4(df, Sub(Get2<0>(x), s), Get2<0>(y), Get2<1>(x), Get2<1>(y)));
}

// Computes x * y in double-float precision
// Translated from common/df.h:107 dfscale_vf2_vf2_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) ScaleDF(const D df, Vec2<D> d, Vec<D> s) {
  return Create2(df, Mul(Get2<0>(d), s), Mul(Get2<1>(d), s));
}

// Computes x + y in double-float precision
// Translated from common/df.h:116 dfadd2_vf2_vf_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) AddDF(const D df, Vec<D> x, Vec<D> y) {
  Vec<D> s = Add(x, y);
  Vec<D> v = Sub(s, x);
  return Create2(df, s, Add(Sub(x, Sub(s, v)), Sub(y, v)));
}

// Computes x + y in double-float precision
// Translated from common/df.h:122 dfadd2_vf2_vf_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) AddDF(const D df, Vec<D> x, Vec2<D> y) {
  Vec<D> s = Add(x, Get2<0>(y));
  Vec<D> v = Sub(s, x);
  return Create2(df, s, Add(Add(Sub(x, Sub(s, v)), Sub(Get2<0>(y), v)), Get2<1>(y)));

}

// Computes x + y in double-float precision, sped up by assuming |x| > |y|
// Translated from common/df.h:111 dfadd_vf2_vf_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) AddFastDF(const D df, Vec<D> x, Vec<D> y) {
  Vec<D> s = Add(x, y);
  return Create2(df, s, Add(Sub(x, s), y));
}

// Computes 1/x in double-float precision
// Translated from common/df.h:219 dfrec_vf2_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) RecDF(const D df, Vec<D> d) {
#if HWY_SLEEF_HAS_FMA
  Vec<D> s = Div(Set(df, 1.0), d);
  return Create2(df, s, Mul(s, NegMulAdd(d, s, Set(df, 1))));
#else
  Vec<D> t = Div(Set(df, 1.0), d);
  Vec<D> dh = LowerPrecision(df, d), dl = Sub(d, dh);
  Vec<D> th = LowerPrecision(df, t), tl = Sub(t, th);

  Vec<D> u = Set(df, -1);
  u = MulAdd(dh, th, u);
  u = MulAdd(dh, tl, u);
  u = MulAdd(dl, th, u);
  u = MulAdd(dl, tl, u);

  return Create2(df, t, Mul(Neg(t), u));
#endif
}

// Extract the sign bit of x into an unsigned integer
// Translated from libm/sleefsimdsp.c:453 vsignbit_vm_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<RebindToUnsigned<D>>) SignBit(const D df, Vec<D> f) {
  RebindToUnsigned<D> du;
  
  return And(BitCast(du, f), BitCast(du, Set(df, -0.0f)));
}

// Calculate x * sign(y) with only bitwise logic
// Translated from libm/sleefsimdsp.c:458 vmulsign_vf_vf_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) MulSignBit(const D df, Vec<D> x, Vec<D> y) {
  RebindToUnsigned<D> du;
  
  return BitCast(df, Xor(BitCast(du, x), SignBit(df, y)));
}

// Integer log of x
// Translated from libm/sleefsimdsp.c:489 vilogbk_vi2_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<RebindToSigned<D>>) ILogB(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Mask<D> o = Lt(d, Set(df, 5.421010862427522E-20f));
  d = IfThenElse(o, Mul(Set(df, 1.8446744073709552E19f), d), d);
  Vec<RebindToSigned<D>> q = And(BitCast(di, ShiftRight<23>(BitCast(du, BitCast(di, d)))), Set(di, 0xff));
  q = Sub(q, IfThenElse(RebindMask(di, o), Set(di, 64 + 0x7f), Set(di, 0x7f)));
  return q;
}

// Specialization of IfThenElse to double-float operands
// Translated from common/df.h:38 vsel_vf2_vo_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) IfThenElse(const D df, Mask<D> m, Vec2<D> x, Vec2<D> y) {
  return Create2(df, IfThenElse(m, Get2<0>(x), Get2<0>(y)), IfThenElse(m, Get2<1>(x), Get2<1>(y)));
}

// Calculate reciprocal square root on ARM NEON platforms
// Translated from arch/helperneon32.h:153 vrecsqrt_vf_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) ReciprocalSqrt(const D df, Vec<D> d) {
  Vec<D> x = Vec<D>{vrsqrteq_f32(d.raw)};
  x = Mul(x, Vec<D>{vrsqrtsq_f32(d.raw, Mul(x, x).raw)});
  return MulAdd(NegMulAdd(x, Mul(x, d), Set(df, 1)), Mul(x, Set(df, 0.5)), x);
}

// Computes sqrt(x) in double-float precision
// Translated from common/df.h:355 dfsqrt_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SqrtDF(const D df, Vec2<D> d) {
  Vec<D> t = Sqrt(Add(Get2<0>(d), Get2<1>(d)));
  return ScaleDF(df, MulDF(df, AddDF(df, d, MulDF(df, t, t)), RecDF(df, t)), Set(df, 0.5));
}

// Computes ln(x) in double-float precision (version 1)
// Translated from libm/sleefsimdsp.c:2198 logkf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) LogDF(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec2<D> x, x2;
  Vec<D> t, m;

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Mask<D> o = Lt(d, Set(df, FloatMin));
  d = IfThenElse(o, Mul(d, Set(df, (float)(INT64_C(1) << 32) * (float)(INT64_C(1) << 32))), d);
  Vec<RebindToSigned<D>> e = ILogB2(df, Mul(d, Set(df, 1.0f/0.75f)));
  m = LoadExp3(df, d, Neg(e));
  e = IfThenElse(RebindMask(di, o), Sub(e, Set(di, 64)), e);
#else
  Vec<D> e = GetExponent(Mul(d, Set(df, 1.0f/0.75f)));
  e = IfThenElse(Eq(e, Inf(df)), Set(df, 128.0f), e);
  m = GetMantissa(d);
#endif

  x = DivDF(df, AddDF(df, Set(df, -1), m), AddDF(df, Set(df, 1), m));
  x2 = SquareDF(df, x);

  t = Set(df, 0.240320354700088500976562);
  t = MulAdd(t, Get2<0>(x2), Set(df, 0.285112679004669189453125));
  t = MulAdd(t, Get2<0>(x2), Set(df, 0.400007992982864379882812));
  Vec2<D> c = Create2(df, Set(df, 0.66666662693023681640625f), Set(df, 3.69183861259614332084311e-09f));

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Vec2<D> s = MulDF(df, Create2(df, Set(df, 0.69314718246459960938f), Set(df, -1.904654323148236017e-09f)), ConvertTo(df, e));
#else
  Vec2<D> s = MulDF(df, Create2(df, Set(df, 0.69314718246459960938f), Set(df, -1.904654323148236017e-09f)), e);
#endif

  s = AddFastDF(df, s, ScaleDF(df, x, Set(df, 2)));
  s = AddFastDF(df, s, MulDF(df, MulDF(df, x2, x),
					     AddDF(df, MulDF(df, x2, t), c)));
  return s;
}

// Create a mask of which is true if x's sign bit is set
// Translated from libm/sleefsimdsp.c:472 vsignbit_vo_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Mask<D>) SignBitMask(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  return RebindMask(df, Eq(And(BitCast(di, d), Set(di, 0x80000000)), Set(di, 0x80000000)));
}

// Sets the exponent of 'x' to 2^e
// Translated from libm/sleefsimdsp.c:520 vldexp_vf_vf_vi2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) LoadExp(const D df, Vec<D> x, Vec<RebindToSigned<D>> q) {
  RebindToSigned<D> di;
  
  Vec<D> u;
  Vec<RebindToSigned<D>> m = ShiftRight<31>(q);
  m = ShiftLeft<4>(Sub(ShiftRight<6>(Add(m, q)), m));
  q = Sub(q, ShiftLeft<2>(m));
  m = Add(m, Set(di, 0x7f));
  m = And(VecFromMask(di, Gt(m, Set(di, 0))), m);
  Vec<RebindToSigned<D>> n = VecFromMask(di, Gt(m, Set(di, 0xff)));
  m = Or(AndNot(n, m), And(n, Set(di, 0xff)));
  u = BitCast(df, ShiftLeft<23>(m));
  x = Mul(Mul(Mul(Mul(x, u), u), u), u);
  u = BitCast(df, ShiftLeft<23>(Add(q, Set(di, 0x7f))));
  return Mul(x, u);
}

// Computes e^x in double-float precision
// Translated from libm/sleefsimdsp.c:2310 expkf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) ExpDF_float(const D df, Vec2<D> d) {
  RebindToUnsigned<D> du;
  
  Vec<D> u = Mul(Add(Get2<0>(d), Get2<1>(d)), Set(df, OneOverLn2_f));
  Vec<RebindToSigned<D>> q = NearestInt(u);
  Vec2<D> s, t;

  s = AddDF(df, d, Mul(ConvertTo(df, q), Set(df, -Ln2Hi_f)));
  s = AddDF(df, s, Mul(ConvertTo(df, q), Set(df, -Ln2Lo_f)));

  s = NormalizeDF(df, s);

  u = Set(df, 0.00136324646882712841033936f);
  u = MulAdd(u, Get2<0>(s), Set(df, 0.00836596917361021041870117f));
  u = MulAdd(u, Get2<0>(s), Set(df, 0.0416710823774337768554688f));
  u = MulAdd(u, Get2<0>(s), Set(df, 0.166665524244308471679688f));
  u = MulAdd(u, Get2<0>(s), Set(df, 0.499999850988388061523438f));

  t = AddFastDF(df, s, MulDF(df, SquareDF(df, s), u));

  t = AddFastDF(df, Set(df, 1), t);
  u = Add(Get2<0>(t), Get2<1>(t));
  u = LoadExp(df, u, q);

  u = BitCast(df, IfThenZeroElse(RebindMask(du, Lt(Get2<0>(d), Set(df, -104))), BitCast(du, u)));
  
  return u;
}

// Bitwise or of x with sign bit of y
// Translated from libm/sleefsimdsp.c:576 vorsign_vf_vf_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) OrSignBit(const D df, Vec<D> x, Vec<D> y) {
  RebindToUnsigned<D> du;
  
  return BitCast(df, Or(BitCast(du, x), SignBit(df, y)));
}

// Helper for Payne Hanek reduction.
// Translated from libm/sleefsimdsp.c:581 rempisubf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) PayneHanekReductionHelper(const D df, Vec<D> x) {
  RebindToSigned<D> di;
  
Vec<D> y = Round(Mul(x, Set(df, 4)));
  Vec<RebindToSigned<D>> vi = ConvertTo(di, Sub(y, Mul(Round(x), Set(df, 4))));
  return Create2(df, Sub(x, Mul(y, Set(df, 0.25))), BitCast(df, vi));

}

// Calculate Payne Hanek reduction. This appears to return ((2*x/pi) - round(2*x/pi)) * pi / 2 and the integer quadrant of x in range -2 to 2 (0 is [-pi/4, pi/4], 2/-2 are from [3pi/4, 5pi/4] with the sign flip a little after pi).
// Translated from libm/sleefsimdsp.c:598 rempif
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec3<D>) PayneHanekReduction(const D df, Vec<D> a) {
  RebindToSigned<D> di;
  
  Vec2<D> x, y;
  Vec<RebindToSigned<D>> ex = ILogB2(df, a);
#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
  ex = AndNot(ShiftRight<31>(ex), ex);
  ex = And(ex, Set(di, 127));
#endif
  ex = Sub(ex, Set(di, 25));
  Vec<RebindToSigned<D>> q = IfThenElseZero(RebindMask(di, RebindMask(df, Gt(ex, Set(di, 90-25)))), Set(di, -64));
  a = LoadExp3(df, a, q);
  ex = AndNot(ShiftRight<31>(ex), ex);
  ex = ShiftLeft<2>(ex);
  x = MulDF(df, a, GatherIndex(df, PayneHanekReductionTable_float, ex));
  Vec2<D> di_ = PayneHanekReductionHelper(df, Get2<0>(x));
  q = BitCast(di, Get2<1>(di_));
  x = Set2<0>(x, Get2<0>(di_));
  x = NormalizeDF(df, x);
  y = MulDF(df, a, GatherIndex(df, PayneHanekReductionTable_float+1, ex));
  x = AddDF(df, x, y);
  di_ = PayneHanekReductionHelper(df, Get2<0>(x));
  q = Add(q, BitCast(di, Get2<1>(di_)));
  x = Set2<0>(x, Get2<0>(di_));
  x = NormalizeDF(df, x);
  y = Create2(df, GatherIndex(df, PayneHanekReductionTable_float+2, ex), GatherIndex(df, PayneHanekReductionTable_float+3, ex));
  y = MulDF(df, y, a);
  x = AddDF(df, x, y);
  x = NormalizeDF(df, x);
  x = MulDF(df, x, Create2(df, Set(df, 3.1415927410125732422f*2), Set(df, -8.7422776573475857731e-08f*2)));
  x = IfThenElse(df, Lt(Abs(a), Set(df, 0.7f)), Create2(df, a, Set(df, 0)), x);
  return Create3(df, Get2<0>(x), Get2<1>(x), BitCast(df, q));
}

// Add (((v0 + 1) + v2) + v3) + v4
// Translated from common/df.h:67 vadd_vf_5vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Add5(const D df, Vec<D> v0, Vec<D> v1, Vec<D> v2, Vec<D> v3, Vec<D> v4) {
  return Add4(df, Add(v0, v1), v2, v3, v4);
}

// Add ((((v0 + 1) + v2) + v3) + v4) + v5
// Translated from common/df.h:71 vadd_vf_6vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Add6(const D df, Vec<D> v0, Vec<D> v1, Vec<D> v2, Vec<D> v3, Vec<D> v4, Vec<D> v5) {
  return Add5(df, Add(v0, v1), v2, v3, v4, v5);
}

// Computes x * y in double-float precision, returning result as single-precision
// Translated from common/df.h:210 dfmul_vf_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) MulDF_float(const D df, Vec2<D> x, Vec2<D> y) {
#if HWY_SLEEF_HAS_FMA
  return MulAdd(Get2<0>(x), Get2<0>(y), MulAdd(Get2<1>(x), Get2<0>(y), Mul(Get2<0>(x), Get2<1>(y))));
#else
  Vec<D> xh = LowerPrecision(df, Get2<0>(x)), xl = Sub(Get2<0>(x), xh);
  Vec<D> yh = LowerPrecision(df, Get2<0>(y)), yl = Sub(Get2<0>(y), yh);

  return Add6(df, Mul(Get2<1>(x), yh), Mul(xh, Get2<1>(y)), Mul(xl, yl), Mul(xh, yl), Mul(xl, yh), Mul(xh, yh));
#endif
}

// Computes 1/x in double-float precision
// Translated from common/df.h:224 dfrec_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) RecDF(const D df, Vec2<D> d) {
#if HWY_SLEEF_HAS_FMA
  Vec<D> s = Div(Set(df, 1.0), Get2<0>(d));
  return Create2(df, s, Mul(s, NegMulAdd(Get2<1>(d), s, NegMulAdd(Get2<0>(d), s, Set(df, 1)))));
#else
  Vec<D> t = Div(Set(df, 1.0), Get2<0>(d));
  Vec<D> dh = LowerPrecision(df, Get2<0>(d)), dl = Sub(Get2<0>(d), dh);
  Vec<D> th = LowerPrecision(df, t  ), tl = Sub(t, th);

  Vec<D> u = Set(df, -1);
  u = MulAdd(dh, th, u);
  u = MulAdd(dh, tl, u);
  u = MulAdd(dl, th, u);
  u = MulAdd(dl, tl, u);
  u = MulAdd(Get2<1>(d), t, u);

  return Create2(df, t, Mul(Neg(t), u));
#endif
}

// Computes x - y in double-float precision, assuming |x| > |y|
// Translated from common/df.h:172 dfsub_vf2_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SubDF(const D df, Vec2<D> x, Vec2<D> y) {
  // |x| >= |y|

  Vec<D> s = Sub(Get2<0>(x), Get2<0>(y));
  Vec<D> t = Sub(Get2<0>(x), s);
  t = Sub(t, Get2<0>(y));
  t = Add(t, Get2<1>(x));
  return Create2(df, s, Sub(t, Get2<1>(y)));
}

// Computes -x in double-float precision
// Translated from common/df.h:93 dfneg_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) NegDF(const D df, Vec2<D> x) {
  return Create2(df, Neg(Get2<0>(x)), Neg(Get2<1>(x)));
}

// Computes e^x - 1 faster with lower precision
// Translated from libm/sleefsimdsp.c:2048 expm1fk
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Expm1Fast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec<RebindToSigned<D>> q = NearestInt(Mul(d, Set(df, OneOverLn2_f)));
  Vec<D> s, u;

  s = MulAdd(ConvertTo(df, q), Set(df, -Ln2Hi_f), d);
  s = MulAdd(ConvertTo(df, q), Set(df, -Ln2Lo_f), s);

  Vec<D> s2 = Mul(s, s), s4 = Mul(s2, s2);
  u = Estrin(s, s2, s4, Set(df, 0.5), Set(df, 0.166666671633720397949219), Set(df, 0.0416664853692054748535156), Set(df, 0.00833336077630519866943359), Set(df, 0.00139304355252534151077271), Set(df, 0.000198527617612853646278381));

  u = MulAdd(Mul(s, s), u, s);

  u = IfThenElse(RebindMask(df, Eq(q, Set(di, 0))), u, Sub(LoadExp2(df, Add(u, Set(df, 1)), q), Set(df, 1)));

  return u;
}

// Computes sqrt(x) in double-float precision
// Translated from common/df.h:366 dfsqrt_vf2_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SqrtDF(const D df, Vec<D> d) {
  Vec<D> t = Sqrt(d);
  return ScaleDF(df, MulDF(df, AddDF(df, d, MulDF(df, t, t)), RecDF(df, t)), Set(df, 0.5f));
}

// Computes x - y in double-float precision, assuming |x| > |y|
// Translated from common/df.h:134 dfsub_vf2_vf2_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SubDF(const D df, Vec2<D> x, Vec<D> y) {
  Vec<D> s = Sub(Get2<0>(x), y);
  return Create2(df, s, Add(Sub(Sub(Get2<0>(x), s), y), Get2<1>(x)));
}

// Computes ln(x) in double-float precision (version 2)
// Translated from libm/sleefsimdsp.c:2526 logk2f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) LogFastDF(const D df, Vec2<D> d) {
  Vec2<D> x, x2, m, s;
  Vec<D> t;
  Vec<RebindToSigned<D>> e;

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  e = ILogB(df, Mul(Get2<0>(d), Set(df, 1.0f/0.75f)));
#else
  e = NearestInt(GetExponent(Mul(Get2<0>(d), Set(df, 1.0f/0.75f))));
#endif
  m = ScaleDF(df, d, Pow2I(df, Neg(e)));

  x = DivDF(df, AddDF(df, m, Set(df, -1)), AddDF(df, m, Set(df, 1)));
  x2 = SquareDF(df, x);

  t = Set(df, 0.2392828464508056640625f);
  t = MulAdd(t, Get2<0>(x2), Set(df, 0.28518211841583251953125f));
  t = MulAdd(t, Get2<0>(x2), Set(df, 0.400005877017974853515625f));
  t = MulAdd(t, Get2<0>(x2), Set(df, 0.666666686534881591796875f));

  s = MulDF(df, Create2(df, Set(df, 0.69314718246459960938f), Set(df, -1.904654323148236017e-09f)), ConvertTo(df, e));
  s = AddFastDF(df, s, ScaleDF(df, x, Set(df, 2)));
  s = AddFastDF(df, s, MulDF(df, MulDF(df, x2, x), t));

  return s;
}

// Zero out x when the sign bit of d is not set
// Translated from libm/sleefsimdsp.c:480 vsel_vi2_vf_vi2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<RebindToSigned<D>>) SignBitOrZero(const D df, Vec<D> d, Vec<RebindToSigned<D>> x) {
  RebindToSigned<D> di;
  
  return IfThenElseZero(RebindMask(di, SignBitMask(df, d)), x);
}

// atan2(x, y) in double-float precision
// Translated from libm/sleefsimdsp.c:1872 atan2kf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) ATan2DF(const D df, Vec2<D> y, Vec2<D> x) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> u;
  Vec2<D> s, t;
  Vec<RebindToSigned<D>> q;
  Mask<D> p;
  Vec<RebindToUnsigned<D>> r;
  
  q = IfThenElse(RebindMask(di, Lt(Get2<0>(x), Set(df, 0))), Set(di, -2), Set(di, 0));
  p = Lt(Get2<0>(x), Set(df, 0));
  r = IfThenElseZero(RebindMask(du, p), BitCast(du, Set(df, -0.0)));
  x = Set2<0>(x, BitCast(df, Xor(BitCast(du, Get2<0>(x)), r)));
  x = Set2<1>(x, BitCast(df, Xor(BitCast(du, Get2<1>(x)), r)));

  q = IfThenElse(RebindMask(di, Lt(Get2<0>(x), Get2<0>(y))), Add(q, Set(di, 1)), q);
  p = Lt(Get2<0>(x), Get2<0>(y));
  s = IfThenElse(df, p, NegDF(df, x), y);
  t = IfThenElse(df, p, y, x);

  s = DivDF(df, s, t);
  t = SquareDF(df, s);
  t = NormalizeDF(df, t);

  u = Set(df, -0.00176397908944636583328247f);
  u = MulAdd(u, Get2<0>(t), Set(df, 0.0107900900766253471374512f));
  u = MulAdd(u, Get2<0>(t), Set(df, -0.0309564601629972457885742f));
  u = MulAdd(u, Get2<0>(t), Set(df, 0.0577365085482597351074219f));
  u = MulAdd(u, Get2<0>(t), Set(df, -0.0838950723409652709960938f));
  u = MulAdd(u, Get2<0>(t), Set(df, 0.109463557600975036621094f));
  u = MulAdd(u, Get2<0>(t), Set(df, -0.142626821994781494140625f));
  u = MulAdd(u, Get2<0>(t), Set(df, 0.199983194470405578613281f));

  t = MulDF(df, t, AddFastDF(df, Set(df, -0.333332866430282592773438f), Mul(u, Get2<0>(t))));
  t = MulDF(df, s, AddFastDF(df, Set(df, 1), t));
  t = AddFastDF(df, MulDF(df, Create2(df, Set(df, 1.5707963705062866211f), Set(df, -4.3711388286737928865e-08f)), ConvertTo(df, q)), t);

  return t;
}

// Computes 2^x, where x is an integer.
// Translated from common/commonfuncs.h:326 vpow2i_vd_vi
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Pow2I(const D df, Vec<RebindToSigned<D>> q) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  q = Add(Set(di, 0x3ff), q);
  Vec<RebindToUnsigned<D>> r = ShiftLeft<32>(BitCast(du, ShiftLeft<20>(q)));
  return BitCast(df, r);
}

// Sets the exponent of 'x' to 2^e. Fast, but "short reach"
// Translated from common/commonfuncs.h:349 vldexp2_vd_vd_vi
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) LoadExp2(const D df, Vec<D> d, Vec<RebindToSigned<D>> e) {
  return Mul(Mul(d, Pow2I(df, ShiftRight<1>(e))), Pow2I(df, Sub(e, ShiftRight<1>(e))));
}

// Normalizes a double-double precision representation (redistributes hi vs. lo value)
// Translated from common/dd.h:108 ddnormalize_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) NormalizeDD(const D df, Vec2<D> t) {
  Vec<D> s = Add(Get2<0>(t), Get2<1>(t));
  return Create2(df, s, Add(Sub(Get2<0>(t), s), Get2<1>(t)));
}

// Add (v0 + 1) + v2
// Translated from common/dd.h:59 vadd_vd_3vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Add3(const D df, Vec<D> v0, Vec<D> v1, Vec<D> v2) {
  return Add(Add(v0, v1), v2);
}

// Computes x + y in double-double precision, sped up by assuming |x| > |y|
// Translated from common/dd.h:147 ddadd_vd2_vd_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) AddFastDD(const D df, Vec<D> x, Vec2<D> y) {
  Vec<D> s = Add(x, Get2<0>(y));
  return Create2(df, s, Add3(df, Sub(x, s), Get2<0>(y), Get2<1>(y)));
}

// Add ((v0 + 1) + v2) + v3
// Translated from common/dd.h:63 vadd_vd_4vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Add4(const D df, Vec<D> v0, Vec<D> v1, Vec<D> v2, Vec<D> v3) {
  return Add3(df, Add(v0, v1), v2, v3);
}

// Add (((v0 + 1) + v2) + v3) + v4
// Translated from common/dd.h:67 vadd_vd_5vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Add5(const D df, Vec<D> v0, Vec<D> v1, Vec<D> v2, Vec<D> v3, Vec<D> v4) {
  return Add4(df, Add(v0, v1), v2, v3, v4);
}

// Set the bottom half of mantissa bits to 0 (used in some double-double math)
// Translated from common/dd.h:33 vupper_vd_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) LowerPrecision(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  
  return BitCast(df, And(BitCast(du, d), Set(du, (static_cast<uint64_t>(0xffffffff) << 32) | 0xf8000000)));
}

// Computes x * y in double-double precision
// Translated from common/dd.h:199 ddmul_vd2_vd_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) MulDD(const D df, Vec<D> x, Vec<D> y) {
#if HWY_SLEEF_HAS_FMA
  Vec<D> s = Mul(x, y);
  return Create2(df, s, MulSub(x, y, s));
#else
  Vec<D> xh = LowerPrecision(df, x), xl = Sub(x, xh);
  Vec<D> yh = LowerPrecision(df, y), yl = Sub(y, yh);

  Vec<D> s = Mul(x, y);
  return Create2(df, s, Add5(df, Mul(xh, yh), Neg(s), Mul(xl, yh), Mul(xh, yl), Mul(xl, yl)));
#endif
}

// Computes x + y in double-double precision
// Translated from common/dd.h:140 ddadd2_vd2_vd2_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) AddDD(const D df, Vec2<D> x, Vec<D> y) {
  Vec<D> s = Add(Get2<0>(x), y);
  Vec<D> v = Sub(s, Get2<0>(x));
  Vec<D> w = Add(Sub(Get2<0>(x), Sub(s, v)), Sub(y, v));
  return Create2(df, s, Add(w, Get2<1>(x)));
}

// Add ((((v0 + 1) + v2) + v3) + v4) + v5
// Translated from common/dd.h:71 vadd_vd_6vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Add6(const D df, Vec<D> v0, Vec<D> v1, Vec<D> v2, Vec<D> v3, Vec<D> v4, Vec<D> v5) {
  return Add5(df, Add(v0, v1), v2, v3, v4, v5);
}

// Add (((((v0 + 1) + v2) + v3) + v4) + v5) + v6
// Translated from common/dd.h:75 vadd_vd_7vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Add7(const D df, Vec<D> v0, Vec<D> v1, Vec<D> v2, Vec<D> v3, Vec<D> v4, Vec<D> v5, Vec<D> v6) {
  return Add6(df, Add(v0, v1), v2, v3, v4, v5, v6);
}

// Computes x * y in double-double precision
// Translated from common/dd.h:209 ddmul_vd2_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) MulDD(const D df, Vec2<D> x, Vec2<D> y) {
#if HWY_SLEEF_HAS_FMA
  Vec<D> s = Mul(Get2<0>(x), Get2<0>(y));
  return Create2(df, s, MulAdd(Get2<0>(x), Get2<1>(y), MulAdd(Get2<1>(x), Get2<0>(y), MulSub(Get2<0>(x), Get2<0>(y), s))));
#else
  Vec<D> xh = LowerPrecision(df, Get2<0>(x)), xl = Sub(Get2<0>(x), xh);
  Vec<D> yh = LowerPrecision(df, Get2<0>(y)), yl = Sub(Get2<0>(y), yh);

  Vec<D> s = Mul(Get2<0>(x), Get2<0>(y));
  return Create2(df, s, Add7(df, Mul(xh, yh), Neg(s), Mul(xl, yh), Mul(xh, yl), Mul(xl, yl), Mul(Get2<0>(x), Get2<1>(y)), Mul(Get2<1>(x), Get2<0>(y))));
#endif
}

// Computes x + y in double-double precision, sped up by assuming |x| > |y|
// Translated from common/dd.h:159 ddadd_vd2_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) AddFastDD(const D df, Vec2<D> x, Vec2<D> y) {
  // |x| >= |y|

  Vec<D> s = Add(Get2<0>(x), Get2<0>(y));
  return Create2(df, s, Add4(df, Sub(Get2<0>(x), s), Get2<0>(y), Get2<1>(x), Get2<1>(y)));
}

// Computes x^2 in double-double precision
// Translated from common/dd.h:204 ddsqu_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SquareDD(const D df, Vec2<D> x) {
#if HWY_SLEEF_HAS_FMA
  Vec<D> s = Mul(Get2<0>(x), Get2<0>(x));
  return Create2(df, s, MulAdd(Add(Get2<0>(x), Get2<0>(x)), Get2<1>(x), MulSub(Get2<0>(x), Get2<0>(x), s)));
#else
  Vec<D> xh = LowerPrecision(df, Get2<0>(x)), xl = Sub(Get2<0>(x), xh);

  Vec<D> s = Mul(Get2<0>(x), Get2<0>(x));
  return Create2(df, s, Add5(df, Mul(xh, xh), Neg(s), Mul(Add(xh, xh), xl), Mul(xl, xl), Mul(Get2<0>(x), Add(Get2<1>(x), Get2<1>(x)))));
#endif
}

// Computes x * y in double-double precision
// Translated from common/dd.h:222 ddmul_vd2_vd2_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) MulDD(const D df, Vec2<D> x, Vec<D> y) {
#if HWY_SLEEF_HAS_FMA
  Vec<D> s = Mul(Get2<0>(x), y);
  return Create2(df, s, MulAdd(Get2<1>(x), y, MulSub(Get2<0>(x), y, s)));
#else
  Vec<D> xh = LowerPrecision(df, Get2<0>(x)), xl = Sub(Get2<0>(x), xh);
  Vec<D> yh = LowerPrecision(df, y  ), yl = Sub(y, yh);

  Vec<D> s = Mul(Get2<0>(x), y);
  return Create2(df, s, Add6(df, Mul(xh, yh), Neg(s), Mul(xl, yh), Mul(xh, yl), Mul(xl, yl), Mul(Get2<1>(x), y)));
#endif
}

// Computes e^x in double-double precision
// Translated from libm/sleefsimddp.c:2397 expk2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) ExpDD(const D df, Vec2<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> u = Mul(Add(Get2<0>(d), Get2<1>(d)), Set(df, OneOverLn2_d));
  Vec<D> dq = Round(u);
  Vec<RebindToSigned<D>> q = ConvertTo(di, Round(dq));
  Vec2<D> s, t;

  s = AddDD(df, d, Mul(dq, Set(df, -Ln2Hi_d)));
  s = AddDD(df, s, Mul(dq, Set(df, -Ln2Lo_d)));

  Vec2<D> s2 = SquareDD(df, s), s4 = SquareDD(df, s2);
  Vec<D> s8 = Mul(Get2<0>(s4), Get2<0>(s4));
  u = Estrin(Get2<0>(s), Get2<0>(s2), Get2<0>(s4), s8, Set(df, +0.4166666666666669905e-1), Set(df, +0.8333333333333347095e-2), Set(df, +0.1388888888886763255e-2), Set(df, +0.1984126984148071858e-3), Set(df, +0.2480158735605815065e-4), Set(df, +0.2755731892386044373e-5), Set(df, +0.2755724800902135303e-6), Set(df, +0.2505230023782644465e-7), Set(df, +0.2092255183563157007e-8), Set(df, +0.1602472219709932072e-9));

  t = AddFastDD(df, Set(df, 0.5), MulDD(df, s, Set(df, +0.1666666666666666574e+0)));
  t = AddFastDD(df, Set(df, 1.0), MulDD(df, t, s));
  t = AddFastDD(df, Set(df, 1.0), MulDD(df, t, s));
  t = AddFastDD(df, t, MulDD(df, s4, u));

  t = Set2<0>(t, LoadExp2(df, Get2<0>(t), q));
  t = Set2<1>(t, LoadExp2(df, Get2<1>(t), q));

  t = Set2<0>(t, BitCast(df, IfThenZeroElse(RebindMask(du, Lt(Get2<0>(d), Set(df, -1000))), BitCast(du, Get2<0>(t)))));
  t = Set2<1>(t, BitCast(df, IfThenZeroElse(RebindMask(du, Lt(Get2<0>(d), Set(df, -1000))), BitCast(du, Get2<1>(t)))));

  return t;
}

// Sub (v0 - 1) - v2
// Translated from common/dd.h:79 vsub_vd_3vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Sub3(const D df, Vec<D> v0, Vec<D> v1, Vec<D> v2) {
  return Sub(Sub(v0, v1), v2);
}

// Sub ((v0 - 1) - v2) - v3
// Translated from common/dd.h:83 vsub_vd_4vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Sub4(const D df, Vec<D> v0, Vec<D> v1, Vec<D> v2, Vec<D> v3) {
  return Sub3(df, Sub(v0, v1), v2, v3);
}

// Sub (((v0 - 1) - v2) - v3) - v4
// Translated from common/dd.h:87 vsub_vd_5vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Sub5(const D df, Vec<D> v0, Vec<D> v1, Vec<D> v2, Vec<D> v3, Vec<D> v4) {
  return Sub4(df, Sub(v0, v1), v2, v3, v4);
}

// Computes x / y in double-double precision
// Translated from common/dd.h:191 dddiv_vd2_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) DivDD(const D df, Vec2<D> n, Vec2<D> d) {
#if HWY_SLEEF_HAS_FMA
  Vec<D> t = Div(Set(df, 1.0), Get2<0>(d));
  Vec<D> s = Mul(Get2<0>(n), t);
  Vec<D> u = MulSub(t, Get2<0>(n), s);
  Vec<D> v = NegMulAdd(Get2<1>(d), t, NegMulAdd(Get2<0>(d), t, Set(df, 1)));
  return Create2(df, s, MulAdd(s, v, MulAdd(Get2<1>(n), t, u)));
#else
  Vec<D> t = Div(Set(df, 1.0), Get2<0>(d));
  Vec<D> dh  = LowerPrecision(df, Get2<0>(d)), dl  = Sub(Get2<0>(d), dh);
  Vec<D> th  = LowerPrecision(df, t  ), tl  = Sub(t, th);
  Vec<D> nhh = LowerPrecision(df, Get2<0>(n)), nhl = Sub(Get2<0>(n), nhh);

  Vec<D> s = Mul(Get2<0>(n), t);

  Vec<D> u = Add5(df, Sub(Mul(nhh, th), s), Mul(nhh, tl), Mul(nhl, th), Mul(nhl, tl),
		    Mul(s, Sub5(df, Set(df, 1), Mul(dh, th), Mul(dh, tl), Mul(dl, th), Mul(dl, tl))));

  return Create2(df, s, MulAdd(t, Sub(Get2<1>(n), Mul(s, Get2<1>(d))), u));
#endif
}

// Sets the exponent of 'x' to 2^e. Very fast, "no denormal"
// Translated from common/commonfuncs.h:353 vldexp3_vd_vd_vi
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) LoadExp3(const D df, Vec<D> d, Vec<RebindToSigned<D>> q) {
  RebindToUnsigned<D> du;
  
  return BitCast(df, Add(BitCast(du, d), ShiftLeft<32>(BitCast(du, ShiftLeft<20>(q)))));
}

// Computes x + y in double-double precision, sped up by assuming |x| > |y|
// Translated from common/dd.h:130 ddadd_vd2_vd2_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) AddFastDD(const D df, Vec2<D> x, Vec<D> y) {
  Vec<D> s = Add(Get2<0>(x), y);
  return Create2(df, s, Add3(df, Sub(Get2<0>(x), s), y, Get2<1>(x)));
}

// Computes x + y in double-double precision
// Translated from common/dd.h:124 ddadd2_vd2_vd_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) AddDD(const D df, Vec<D> x, Vec<D> y) {
  Vec<D> s = Add(x, y);
  Vec<D> v = Sub(s, x);
  return Create2(df, s, Add(Sub(x, Sub(s, v)), Sub(y, v)));
}

// Integer log of x, "but the argument must be a normalized value"
// Translated from common/commonfuncs.h:300 vilogb2k_vi_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<RebindToSigned<D>>) ILogB2(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<RebindToSigned<D>> q = BitCast(di, ShiftRight<32>(BitCast(du, d)));
  q = BitCast(di, ShiftRight<20>(BitCast(du, q)));
  q = And(q, Set(di, 0x7ff));
  q = Sub(q, Set(di, 0x3ff));
  return q;
}

// Computes x * y in double-double precision
// Translated from common/dd.h:113 ddscale_vd2_vd2_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) ScaleDD(const D df, Vec2<D> d, Vec<D> s) {
  return Create2(df, Mul(Get2<0>(d), s), Mul(Get2<1>(d), s));
}

// Computes x + y in double-double precision
// Translated from common/dd.h:152 ddadd2_vd2_vd_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) AddDD(const D df, Vec<D> x, Vec2<D> y) {
  Vec<D> s = Add(x, Get2<0>(y));
  Vec<D> v = Sub(s, x);
  return Create2(df, s, Add(Add(Sub(x, Sub(s, v)), Sub(Get2<0>(y), v)), Get2<1>(y)));
}

// Computes x + y in double-double precision, sped up by assuming |x| > |y|
// Translated from common/dd.h:119 ddadd_vd2_vd_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) AddFastDD(const D df, Vec<D> x, Vec<D> y) {
  Vec<D> s = Add(x, y);
  return Create2(df, s, Add(Sub(x, s), y));
}

// Computes 1/x in double-double precision
// Translated from common/dd.h:227 ddrec_vd2_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) RecDD(const D df, Vec<D> d) {
#if HWY_SLEEF_HAS_FMA
  Vec<D> s = Div(Set(df, 1.0), d);
  return Create2(df, s, Mul(s, NegMulAdd(d, s, Set(df, 1))));
#else
  Vec<D> t = Div(Set(df, 1.0), d);
  Vec<D> dh = LowerPrecision(df, d), dl = Sub(d, dh);
  Vec<D> th = LowerPrecision(df, t), tl = Sub(t, th);

  return Create2(df, t, Mul(t, Sub5(df, Set(df, 1), Mul(dh, th), Mul(dh, tl), Mul(dl, th), Mul(dl, tl))));
#endif
}

// Integer log of x
// Translated from common/commonfuncs.h:290 vilogbk_vi_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<RebindToSigned<D>>) ILogB(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Mask<D> o = Lt(d, Set(df, 4.9090934652977266E-91));
  d = IfThenElse(o, Mul(Set(df, 2.037035976334486E90), d), d);
  Vec<RebindToSigned<D>> q = BitCast(di, ShiftRight<32>(BitCast(du, d)));
  q = And(q, Set(di, (int)(((1U << 12) - 1) << 20)));
  q = BitCast(di, ShiftRight<20>(BitCast(du, q)));
  q = Sub(q, IfThenElse(RebindMask(di, o), Set(di, 300 + 0x3ff), Set(di, 0x3ff)));
  return q;
}

// Extract the sign bit of x into an unsigned integer
// Translated from common/commonfuncs.h:196 vsignbit_vm_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<RebindToUnsigned<D>>) SignBit(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  
  return And(BitCast(du, d), BitCast(du, Set(df, -0.0)));
}

// Calculate x * sign(y) with only bitwise logic
// Translated from common/commonfuncs.h:214 vmulsign_vd_vd_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) MulSignBit(const D df, Vec<D> x, Vec<D> y) {
  RebindToUnsigned<D> du;
  
  return BitCast(df, Xor(BitCast(du, x), SignBit(df, y)));
}

// Specialization of IfThenElse to double-double operands
// Translated from common/dd.h:49 vsel_vd2_vo_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) IfThenElse(const D df, Mask<D> m, Vec2<D> x, Vec2<D> y) {
  return Create2(df, IfThenElse(m, Get2<0>(x), Get2<0>(y)), IfThenElse(m, Get2<1>(x), Get2<1>(y)));
}

// Computes x + y in double-double precision
// Translated from common/dd.h:166 ddadd2_vd2_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) AddDD(const D df, Vec2<D> x, Vec2<D> y) {
  Vec<D> s = Add(Get2<0>(x), Get2<0>(y));
  Vec<D> v = Sub(s, Get2<0>(x));
  Vec<D> t = Add(Sub(Get2<0>(x), Sub(s, v)), Sub(Get2<0>(y), v));
  return Create2(df, s, Add(t, Add(Get2<1>(x), Get2<1>(y))));
}

// Computes sqrt(x) in double-double precision
// Translated from common/dd.h:312 ddsqrt_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SqrtDD(const D df, Vec2<D> d) {
  Vec<D> t = Sqrt(Add(Get2<0>(d), Get2<1>(d)));
  return ScaleDD(df, MulDD(df, AddDD(df, d, MulDD(df, t, t)), RecDD(df, t)), Set(df, 0.5));
}

// Bitwise or of x with sign bit of y
// Translated from common/commonfuncs.h:224 vorsign_vd_vd_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) OrSignBit(const D df, Vec<D> x, Vec<D> y) {
  RebindToUnsigned<D> du;
  
  return BitCast(df, Or(BitCast(du, x), SignBit(df, y)));
}

// True if d is an odd (assuming d is an integer)
// Translated from common/commonfuncs.h:282 visodd_vo_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Mask<D>) IsOdd(const D df, Vec<D> d) {
  Vec<D> x = Mul(d, Set(df, 0.5));
  return Ne(Round(x), x);
}

// True if d is an integer
// Translated from common/commonfuncs.h:278 visint_vo_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Mask<D>) IsInt(const D df, Vec<D> d) {
  return Eq(Round(d), d);
}

// Computes ln(x) in double-double precision (version 1)
// Translated from libm/sleefsimddp.c:2223 logk
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) LogDD(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec2<D> x, x2, s;
  Vec<D> t, m;

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Mask<D> o = Lt(d, Set(df, DoubleMin));
  d = IfThenElse(o, Mul(d, Set(df, (double)(INT64_C(1) << 32) * (double)(INT64_C(1) << 32))), d);
  Vec<RebindToSigned<D>> e = ILogB2(df, Mul(d, Set(df, 1.0/0.75)));
  m = LoadExp3(df, d, Neg(e));
  e = IfThenElse(RebindMask(di, o), Sub(e, Set(di, 64)), e);
#else
  Vec<D> e = GetExponent(Mul(d, Set(df, 1.0/0.75)));
  e = IfThenElse(Eq(e, Inf(df)), Set(df, 1024.0), e);
  m = GetMantissa(d);
#endif

  x = DivDD(df, AddDD(df, Set(df, -1), m), AddDD(df, Set(df, 1), m));
  x2 = SquareDD(df, x);

  Vec<D> x4 = Mul(Get2<0>(x2), Get2<0>(x2)), x8 = Mul(x4, x4), x16 = Mul(x8, x8);
  t = Estrin(Get2<0>(x2), x4, x8, x16, Set(df, 0.400000000000000077715612), Set(df, 0.285714285714249172087875), Set(df, 0.222222222230083560345903), Set(df, 0.181818180850050775676507), Set(df, 0.153846227114512262845736), Set(df, 0.13332981086846273921509), Set(df, 0.117754809412463995466069), Set(df, 0.103239680901072952701192), Set(df, 0.116255524079935043668677));

  Vec2<D> c = Create2(df, Set(df, 0.666666666666666629659233), Set(df, 3.80554962542412056336616e-17));
#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  s = MulDD(df, Create2(df, Set(df, 0.693147180559945286226764), Set(df, 2.319046813846299558417771e-17)), ConvertTo(df, e));
#else
  s = MulDD(df, Create2(df, Set(df, 0.693147180559945286226764), Set(df, 2.319046813846299558417771e-17)), e);
#endif
  s = AddFastDD(df, s, ScaleDD(df, x, Set(df, 2)));
  x = MulDD(df, x2, x);
  s = AddFastDD(df, s, MulDD(df, x, c));
  x = MulDD(df, x2, x);
  s = AddFastDD(df, s, MulDD(df, x, t));

  return s;
}

// Create a mask of which is true if x's sign bit is set
// Translated from common/commonfuncs.h:200 vsignbit_vo_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Mask<D>) SignBitMask(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  
  return RebindMask(df, Eq(And(BitCast(du, d), BitCast(du, Set(df, -0.0))), BitCast(du, Set(df, -0.0))));
}

// Computes e^x in double-double precision
// Translated from libm/sleefsimddp.c:2322 expk
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) ExpDD_double(const D df, Vec2<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> u = Mul(Add(Get2<0>(d), Get2<1>(d)), Set(df, OneOverLn2_d));
  Vec<D> dq = Round(u);
  Vec<RebindToSigned<D>> q = ConvertTo(di, Round(dq));
  Vec2<D> s, t;

  s = AddDD(df, d, Mul(dq, Set(df, -Ln2Hi_d)));
  s = AddDD(df, s, Mul(dq, Set(df, -Ln2Lo_d)));

  s = NormalizeDD(df, s);

  Vec<D> s2 = Mul(Get2<0>(s), Get2<0>(s)), s4 = Mul(s2, s2), s8 = Mul(s4, s4);
  u = Estrin(Get2<0>(s), s2, s4, s8, Set(df, 0.500000000000000999200722), Set(df, 0.166666666666666740681535), Set(df, 0.0416666666665409524128449), Set(df, 0.00833333333332371417601081), Set(df, 0.0013888888939977128960529), Set(df, 0.000198412698809069797676111), Set(df, 2.48014973989819794114153e-05), Set(df, 2.75572496725023574143864e-06), Set(df, 2.76286166770270649116855e-07), Set(df, 2.51069683420950419527139e-08));

  t = AddFastDD(df, Set(df, 1), s);
  t = AddFastDD(df, t, MulDD(df, SquareDD(df, s), u));

  u = Add(Get2<0>(t), Get2<1>(t));
  u = LoadExp2(df, u, q);

  u = BitCast(df, IfThenZeroElse(RebindMask(du, Lt(Get2<0>(d), Set(df, -1000))), BitCast(du, u)));
  
  return u;
}

// Computes x * y in double-double precision, returning result as single-precision
// Translated from common/dd.h:214 ddmul_vd_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) MulDD_double(const D df, Vec2<D> x, Vec2<D> y) {
#if HWY_SLEEF_HAS_FMA
  return MulAdd(Get2<0>(x), Get2<0>(y), MulAdd(Get2<1>(x), Get2<0>(y), Mul(Get2<0>(x), Get2<1>(y))));
#else
  Vec<D> xh = LowerPrecision(df, Get2<0>(x)), xl = Sub(Get2<0>(x), xh);
  Vec<D> yh = LowerPrecision(df, Get2<0>(y)), yl = Sub(Get2<0>(y), yh);

  return Add6(df, Mul(Get2<1>(x), yh), Mul(xh, Get2<1>(y)), Mul(xl, yl), Mul(xh, yl), Mul(xl, yh), Mul(xh, yh));
#endif
}

// Helper for Payne Hanek reduction.
// Translated from common/commonfuncs.h:423 rempisub
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) PayneHanekReductionHelper_d(const D df, Vec<D> x) {
  RebindToSigned<D> di;
  
Vec<D> y = Round(Mul(x, Set(df, 4)));
  Vec<RebindToSigned<D>> vi = ConvertTo(di, Trunc(Sub(y, Mul(Round(x), Set(df, 4)))));
  return Create2(df, Sub(x, Mul(y, Set(df, 0.25))), BitCast(df, vi));

}

// Calculate Payne Hanek reduction. This appears to return ((2*x/pi) - round(2*x/pi)) * pi / 2 and the integer quadrant of x in range -2 to 2 (0 is [-pi/4, pi/4], 2/-2 are from [3pi/4, 5pi/4] with the sign flip a little after pi).
// Translated from libm/sleefsimddp.c:348 rempi
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec3<D>) PayneHanekReduction_d(const D df, Vec<D> a) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec2<D> x, y;
  Vec<RebindToSigned<D>> ex = ILogB2(df, a);
#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
  ex = AndNot(ShiftRight<31>(ex), ex);
  ex = And(ex, Set(di, 1023));
#endif
  ex = Sub(ex, Set(di, 55));
  Vec<RebindToSigned<D>> q = IfThenElseZero(RebindMask(di, RebindMask(df, Gt(ex, Set(di, 700-55)))), Set(di, -64));
  a = LoadExp3(df, a, q);
  ex = AndNot(ShiftRight<31>(ex), ex);
  ex = ShiftLeft<2>(ex);
  x = MulDD(df, a, GatherIndex(df, PayneHanekReductionTable_double, ex));
  Vec2<D> di_ = PayneHanekReductionHelper_d(df, Get2<0>(x));
  q = BitCast(di, Get2<1>(di_));
  x = Set2<0>(x, Get2<0>(di_));
  x = NormalizeDD(df, x);
  y = MulDD(df, a, GatherIndex(df, PayneHanekReductionTable_double+1, ex));
  x = AddDD(df, x, y);
  di_ = PayneHanekReductionHelper_d(df, Get2<0>(x));
  q = Add(q, BitCast(di, Get2<1>(di_)));
  x = Set2<0>(x, Get2<0>(di_));
  x = NormalizeDD(df, x);
  y = Create2(df, GatherIndex(df, PayneHanekReductionTable_double+2, ex), GatherIndex(df, PayneHanekReductionTable_double+3, ex));
  y = MulDD(df, y, a);
  x = AddDD(df, x, y);
  x = NormalizeDD(df, x);
  x = MulDD(df, x, Create2(df, Set(df, 3.141592653589793116*2), Set(df, 1.2246467991473532072e-16*2)));
  Mask<D> o = Lt(Abs(a), Set(df, 0.7));
  x = Set2<0>(x, IfThenElse(o, a, Get2<0>(x)));
  x = Set2<1>(x, BitCast(df, IfThenZeroElse(RebindMask(du, o), BitCast(du, Get2<1>(x)))));
  return Create3(df, Get2<0>(x), Get2<1>(x), BitCast(df, q));
}

// Compmutes -x in double-double precision
// Translated from common/dd.h:97 ddneg_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) NegDD(const D df, Vec2<D> x) {
  return Create2(df, Neg(Get2<0>(x)), Neg(Get2<1>(x)));
}

// Computes x - y in double-double precision, assuming |x| > |y|
// Translated from common/dd.h:180 ddsub_vd2_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SubDD(const D df, Vec2<D> x, Vec2<D> y) {
  // |x| >= |y|

  Vec<D> s = Sub(Get2<0>(x), Get2<0>(y));
  Vec<D> t = Sub(Get2<0>(x), s);
  t = Sub(t, Get2<0>(y));
  t = Add(t, Get2<1>(x));
  return Create2(df, s, Sub(t, Get2<1>(y)));
}

// Sub ((((v0 - 1) - v2) - v3) - v4) - v5
// Translated from common/dd.h:91 vsub_vd_6vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Sub6(const D df, Vec<D> v0, Vec<D> v1, Vec<D> v2, Vec<D> v3, Vec<D> v4, Vec<D> v5) {
  return Sub5(df, Sub(v0, v1), v2, v3, v4, v5);
}

// Computes 1/x in double-double precision
// Translated from common/dd.h:232 ddrec_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) RecDD(const D df, Vec2<D> d) {
#if HWY_SLEEF_HAS_FMA
  Vec<D> s = Div(Set(df, 1.0), Get2<0>(d));
  return Create2(df, s, Mul(s, NegMulAdd(Get2<1>(d), s, NegMulAdd(Get2<0>(d), s, Set(df, 1)))));
#else
  Vec<D> t = Div(Set(df, 1.0), Get2<0>(d));
  Vec<D> dh = LowerPrecision(df, Get2<0>(d)), dl = Sub(Get2<0>(d), dh);
  Vec<D> th = LowerPrecision(df, t  ), tl = Sub(t, th);

  return Create2(df, t, Mul(t, Sub6(df, Set(df, 1), Mul(dh, th), Mul(dh, tl), Mul(dl, th), Mul(dl, tl), Mul(Get2<1>(d), t))));
#endif
}

// Computes e^x - 1
// Translated from libm/sleefsimddp.c:2195 expm1k
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Expm1k(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec<D> u = Round(Mul(d, Set(df, OneOverLn2_d))), s;
  Vec<RebindToSigned<D>> q = ConvertTo(di, Round(u));

  s = MulAdd(u, Set(df, -Ln2Hi_d), d);
  s = MulAdd(u, Set(df, -Ln2Lo_d), s);

  Vec<D> s2 = Mul(s, s), s4 = Mul(s2, s2), s8 = Mul(s4, s4);
  u = Estrin(s, s2, s4, s8, Set(df, 0.166666666666666851703837), Set(df, 0.0416666666666665047591422), Set(df, 0.00833333333331652721664984), Set(df, 0.00138888888889774492207962), Set(df, 0.000198412698960509205564975), Set(df, 2.4801587159235472998791e-05), Set(df, 2.75572362911928827629423e-06), Set(df, 2.75573911234900471893338e-07), Set(df, 2.51112930892876518610661e-08), Set(df, 2.08860621107283687536341e-09));

  u = Add(MulAdd(s2, Set(df, 0.5), Mul(Mul(s2, s), u)), s);
  
  u = IfThenElse(RebindMask(df, Eq(q, Set(di, 0))), u, Sub(LoadExp2(df, Add(u, Set(df, 1)), q), Set(df, 1)));

  return u;
}

// Computes sqrt(x) in double-double precision
// Translated from common/dd.h:317 ddsqrt_vd2_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SqrtDD(const D df, Vec<D> d) {
  Vec<D> t = Sqrt(d);
  return ScaleDD(df, MulDD(df, AddDD(df, d, MulDD(df, t, t)), RecDD(df, t)), Set(df, 0.5));
}

// Computes x - y in double-double precision, assuming |x| > |y|
// Translated from common/dd.h:135 ddsub_vd2_vd2_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SubDD(const D df, Vec2<D> x, Vec<D> y) {
  Vec<D> s = Sub(Get2<0>(x), y);
  return Create2(df, s, Add(Sub(Sub(Get2<0>(x), s), y), Get2<1>(x)));
}

// Computes ln(x) in double-double precision (version 2)
// Translated from libm/sleefsimddp.c:2508 logk2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) LogFastDD(const D df, Vec2<D> d) {
  Vec2<D> x, x2, m, s;
  Vec<D> t;
  Vec<RebindToSigned<D>> e;
  
  e = ILogB(df, Mul(Get2<0>(d), Set(df, 1.0/0.75)));

  m = Create2(df, LoadExp2(df, Get2<0>(d), Neg(e)), LoadExp2(df, Get2<1>(d), Neg(e)));

  x = DivDD(df, AddDD(df, m, Set(df, -1)), AddDD(df, m, Set(df, 1)));
  x2 = SquareDD(df, x);

  Vec<D> x4 = Mul(Get2<0>(x2), Get2<0>(x2)), x8 = Mul(x4, x4);
  t = Estrin(Get2<0>(x2), x4, x8, Set(df, 0.400000000000914013309483), Set(df, 0.285714285511134091777308), Set(df, 0.22222224632662035403996), Set(df, 0.181816523941564611721589), Set(df, 0.153914168346271945653214), Set(df, 0.131699838841615374240845), Set(df, 0.13860436390467167910856));
  t = MulAdd(t, Get2<0>(x2), Set(df, 0.666666666666664853302393));

  s = MulDD(df, Create2(df, Set(df, 0.693147180559945286226764), Set(df, 2.319046813846299558417771e-17)), ConvertTo(df, e));
  s = AddFastDD(df, s, ScaleDD(df, x, Set(df, 2)));
  s = AddFastDD(df, s, MulDD(df, MulDD(df, x2, x), t));

  return  s;
}

// Zero out x when the sign bit of d is not set
// Translated from libm/sleefsimddp.c:334 vsel_vi_vd_vi
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<RebindToSigned<D>>) SignBitOrZero(const D df, Vec<D> d, Vec<RebindToSigned<D>> x) {
  RebindToSigned<D> di;
  
 return IfThenElseZero(RebindMask(di, SignBitMask(df, d)), x); }

// return d0 < d1 ? x : y
// Translated from libm/sleefsimddp.c:331 vsel_vi_vd_vd_vi_vi
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<RebindToSigned<D>>) LessThanSelect(const D df, Vec<D> d0, Vec<D> d1, Vec<RebindToSigned<D>> x, Vec<RebindToSigned<D>> y) {
  RebindToSigned<D> di;
  
 return IfThenElse(RebindMask(di, Lt(d0, d1)), x, y); }

// atan2(x, y) in double-double precision
// Translated from libm/sleefsimddp.c:1835 atan2k_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) ATan2DD(const D df, Vec2<D> y, Vec2<D> x) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> u;
  Vec2<D> s, t;
  Vec<RebindToSigned<D>> q;
  Mask<D> p;

  q = SignBitOrZero(df, Get2<0>(x), Set(di, -2));
  p = Lt(Get2<0>(x), Set(df, 0));
  Vec<RebindToUnsigned<D>> b = IfThenElseZero(RebindMask(du, p), BitCast(du, Set(df, -0.0)));
  x = Set2<0>(x, BitCast(df, Xor(b, BitCast(du, Get2<0>(x)))));
  x = Set2<1>(x, BitCast(df, Xor(b, BitCast(du, Get2<1>(x)))));

  q = LessThanSelect(df, Get2<0>(x), Get2<0>(y), Add(q, Set(di, 1)), q);
  p = Lt(Get2<0>(x), Get2<0>(y));
  s = IfThenElse(df, p, NegDD(df, x), y);
  t = IfThenElse(df, p, y, x);

  s = DivDD(df, s, t);
  t = SquareDD(df, s);
  t = NormalizeDD(df, t);

  Vec<D> t2 = Mul(Get2<0>(t), Get2<0>(t)), t4 = Mul(t2, t2), t8 = Mul(t4, t4);
  u = Estrin(Get2<0>(t), t2, t4, t8, Set(df, -0.0909090442773387574781907), Set(df, 0.0769225330296203768654095), Set(df, -0.0666620884778795497194182), Set(df, 0.0587946590969581003860434), Set(df, -0.0524914210588448421068719), Set(df, 0.0470843011653283988193763), Set(df, -0.041848579703592507506027), Set(df, 0.0359785005035104590853656), Set(df, -0.0289002344784740315686289), Set(df, 0.0208024799924145797902497), Set(df, -0.0128281333663399031014274), Set(df, 0.00646262899036991172313504), Set(df, -0.00251865614498713360352999), Set(df, 0.00070557664296393412389774), Set(df, -0.000125620649967286867384336), Set(df, 1.06298484191448746607415e-05));
  u = MulAdd(u, Get2<0>(t), Set(df, 0.111111108376896236538123));
  u = MulAdd(u, Get2<0>(t), Set(df, -0.142857142756268568062339));
  u = MulAdd(u, Get2<0>(t), Set(df, 0.199999999997977351284817));
  u = MulAdd(u, Get2<0>(t), Set(df, -0.333333333333317605173818));

  t = AddFastDD(df, s, MulDD(df, MulDD(df, s, t), u));
  
  t = AddFastDD(df, MulDD(df, Create2(df, Set(df, 1.570796326794896557998982), Set(df, 6.12323399573676603586882e-17)), ConvertTo(df, q)), t);

  return t;
}

}

// Computes e^x
// Translated from libm/sleefsimdsp.c:2023 xexpf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Exp(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  
  Vec<RebindToSigned<D>> q = NearestInt(Mul(d, Set(df, OneOverLn2_f)));
  Vec<D> s, u;

  s = MulAdd(ConvertTo(df, q), Set(df, -Ln2Hi_f), d);
  s = MulAdd(ConvertTo(df, q), Set(df, -Ln2Lo_f), s);

  u = Set(df, 0.000198527617612853646278381);
  u = MulAdd(u, s, Set(df, 0.00139304355252534151077271));
  u = MulAdd(u, s, Set(df, 0.00833336077630519866943359));
  u = MulAdd(u, s, Set(df, 0.0416664853692054748535156));
  u = MulAdd(u, s, Set(df, 0.166666671633720397949219));
  u = MulAdd(u, s, Set(df, 0.5));

  u = Add(Set(df, 1.0f), MulAdd(Mul(s, s), u, s));

  u = LoadExp2(df, u, q);

  u = BitCast(df, IfThenZeroElse(RebindMask(du, Lt(d, Set(df, -104))), BitCast(du, u)));
  u = IfThenElse(Lt(Set(df, 100), d), Set(df, InfFloat), u);

  return u;
}

// Computes 2^x
// Translated from libm/sleefsimdsp.c:2605 xexp2f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Exp2(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  
  Vec<D> u = Round(d), s;
  Vec<RebindToSigned<D>> q = NearestInt(u);

  s = Sub(d, u);

  u = Set(df, +0.1535920892e-3);
  u = MulAdd(u, s, Set(df, +0.1339262701e-2));
  u = MulAdd(u, s, Set(df, +0.9618384764e-2));
  u = MulAdd(u, s, Set(df, +0.5550347269e-1));
  u = MulAdd(u, s, Set(df, +0.2402264476e+0));
  u = MulAdd(u, s, Set(df, +0.6931471825e+0));

#ifdef HWY_SLEEF_HAS_FMA
  u = MulAdd(u, s, Set(df, 1));
#else
  u = Get2<0>(NormalizeDF(df, AddFastDF(df, Set(df, 1), MulDF(df, u, s))));
#endif
  
  u = LoadExp2(df, u, q);

  u = IfThenElse(Ge(d, Set(df, 128)), Set(df, InfDouble), u);
  u = BitCast(df, IfThenZeroElse(RebindMask(du, Lt(d, Set(df, -150))), BitCast(du, u)));

  return u;
}

// Computes 10^x
// Translated from libm/sleefsimdsp.c:2654 xexp10f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Exp10(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  
  Vec<D> u = Round(Mul(d, Set(df, Lg10))), s;
  Vec<RebindToSigned<D>> q = NearestInt(u);

  s = MulAdd(u, Set(df, -Log10Of2_Hi_f), d);
  s = MulAdd(u, Set(df, -Log10Of2_Lo_f), s);

  u = Set(df, +0.6802555919e-1);
  u = MulAdd(u, s, Set(df, +0.2078080326e+0));
  u = MulAdd(u, s, Set(df, +0.5393903852e+0));
  u = MulAdd(u, s, Set(df, +0.1171245337e+1));
  u = MulAdd(u, s, Set(df, +0.2034678698e+1));
  u = MulAdd(u, s, Set(df, +0.2650949001e+1));
  Vec2<D> x = AddFastDF(df, Create2(df, Set(df, 2.3025851249694824219), Set(df, -3.1705172516493593157e-08)), Mul(u, s));
  u = Get2<0>(NormalizeDF(df, AddFastDF(df, Set(df, 1), MulDF(df, x, s))));
  
  u = LoadExp2(df, u, q);

  u = IfThenElse(Gt(d, Set(df, 38.5318394191036238941387f)), Set(df, InfFloat), u);
  u = BitCast(df, IfThenZeroElse(RebindMask(du, Lt(d, Set(df, -50))), BitCast(du, u)));

  return u;
}

// Computes e^x - 1
// Translated from libm/sleefsimdsp.c:2701 xexpm1f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Expm1(const D df, Vec<D> a) {
  Vec2<D> d = AddDF(df, ExpDF(df, Create2(df, a, Set(df, 0))), Set(df, -1.0));
  Vec<D> x = Add(Get2<0>(d), Get2<1>(d));
  x = IfThenElse(Gt(a, Set(df, 88.72283172607421875f)), Set(df, InfFloat), x);
  x = IfThenElse(Lt(a, Set(df, -16.635532333438687426013570f)), Set(df, -1), x);
  x = IfThenElse(Eq(a, Set(df, -0.0)), Set(df, -0.0f), x);
  return x;
}

// Computes ln(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2268 xlogf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Log(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec2<D> x;
  Vec<D> t, m, x2;

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Mask<D> o = Lt(d, Set(df, FloatMin));
  d = IfThenElse(o, Mul(d, Set(df, (float)(INT64_C(1) << 32) * (float)(INT64_C(1) << 32))), d);
  Vec<RebindToSigned<D>> e = ILogB2(df, Mul(d, Set(df, 1.0f/0.75f)));
  m = LoadExp3(df, d, Neg(e));
  e = IfThenElse(RebindMask(di, o), Sub(e, Set(di, 64)), e);
  Vec2<D> s = MulDF(df, Create2(df, Set(df, 0.69314718246459960938f), Set(df, -1.904654323148236017e-09f)), ConvertTo(df, e));
#else
  Vec<D> e = GetExponent(Mul(d, Set(df, 1.0f/0.75f)));
  e = IfThenElse(Eq(e, Inf(df)), Set(df, 128.0f), e);
  m = GetMantissa(d);
  Vec2<D> s = MulDF(df, Create2(df, Set(df, 0.69314718246459960938f), Set(df, -1.904654323148236017e-09f)), e);
#endif

  x = DivDF(df, AddDF(df, Set(df, -1), m), AddDF(df, Set(df, 1), m));
  x2 = Mul(Get2<0>(x), Get2<0>(x));

  t = Set(df, +0.3027294874e+0f);
  t = MulAdd(t, x2, Set(df, +0.3996108174e+0f));
  t = MulAdd(t, x2, Set(df, +0.6666694880e+0f));
  
  s = AddFastDF(df, s, ScaleDF(df, x, Set(df, 2)));
  s = AddFastDF(df, s, Mul(Mul(x2, Get2<0>(x)), t));

  Vec<D> r = Add(Get2<0>(s), Get2<1>(s));

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  r = IfThenElse(Eq(d, Inf(df)), Set(df, InfFloat), r);
  r = IfThenElse(Or(Lt(d, Set(df, 0)), IsNaN(d)), Set(df, NanFloat), r);
  r = IfThenElse(Eq(d, Set(df, 0)), Set(df, -InfFloat), r);
#else
  r = Fixup<0>(r, d, Set(di, (4 << (2*4)) | (3 << (4*4)) | (5 << (5*4)) | (2 << (6*4))));
#endif
  
  return r;
}

// Computes ln(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1984 xlogf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) LogFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec<D> x, x2, t, m;

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Mask<D> o = Lt(d, Set(df, FloatMin));
  d = IfThenElse(o, Mul(d, Set(df, (float)(INT64_C(1) << 32) * (float)(INT64_C(1) << 32))), d);
  Vec<RebindToSigned<D>> e = ILogB2(df, Mul(d, Set(df, 1.0f/0.75f)));
  m = LoadExp3(df, d, Neg(e));
  e = IfThenElse(RebindMask(di, o), Sub(e, Set(di, 64)), e);
#else
  Vec<D> e = GetExponent(Mul(d, Set(df, 1.0f/0.75f)));
  e = IfThenElse(Eq(e, Inf(df)), Set(df, 128.0f), e);
  m = GetMantissa(d);
#endif
  
  x = Div(Sub(m, Set(df, 1.0f)), Add(Set(df, 1.0f), m));
  x2 = Mul(x, x);

  t = Set(df, 0.2392828464508056640625f);
  t = MulAdd(t, x2, Set(df, 0.28518211841583251953125f));
  t = MulAdd(t, x2, Set(df, 0.400005877017974853515625f));
  t = MulAdd(t, x2, Set(df, 0.666666686534881591796875f));
  t = MulAdd(t, x2, Set(df, 2.0f));

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  x = MulAdd(x, t, Mul(Set(df, 0.693147180559945286226764f), ConvertTo(df, e)));
  x = IfThenElse(Eq(d, Inf(df)), Set(df, InfFloat), x);
  x = IfThenElse(Or(Lt(d, Set(df, 0)), IsNaN(d)), Set(df, NanFloat), x);
  x = IfThenElse(Eq(d, Set(df, 0)), Set(df, -InfFloat), x);
#else
  x = MulAdd(x, t, Mul(Set(df, 0.693147180559945286226764f), e));
  x = Fixup<0>(x, d, Set(di, (5 << (5*4))));
#endif
  
  return x;
}

// Computes log10(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2712 xlog10f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Log10(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec2<D> x;
  Vec<D> t, m, x2;

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Mask<D> o = Lt(d, Set(df, FloatMin));
  d = IfThenElse(o, Mul(d, Set(df, (float)(INT64_C(1) << 32) * (float)(INT64_C(1) << 32))), d);
  Vec<RebindToSigned<D>> e = ILogB2(df, Mul(d, Set(df, 1.0/0.75)));
  m = LoadExp3(df, d, Neg(e));
  e = IfThenElse(RebindMask(di, o), Sub(e, Set(di, 64)), e);
#else
  Vec<D> e = GetExponent(Mul(d, Set(df, 1.0/0.75)));
  e = IfThenElse(Eq(e, Inf(df)), Set(df, 128.0f), e);
  m = GetMantissa(d);
#endif

  x = DivDF(df, AddDF(df, Set(df, -1), m), AddDF(df, Set(df, 1), m));
  x2 = Mul(Get2<0>(x), Get2<0>(x));

  t = Set(df, +0.1314289868e+0);
  t = MulAdd(t, x2, Set(df, +0.1735493541e+0));
  t = MulAdd(t, x2, Set(df, +0.2895309627e+0));
  
#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Vec2<D> s = MulDF(df, Create2(df, Set(df, 0.30103001), Set(df, -1.432098889e-08)), ConvertTo(df, e));
#else
  Vec2<D> s = MulDF(df, Create2(df, Set(df, 0.30103001), Set(df, -1.432098889e-08)), e);
#endif

  s = AddFastDF(df, s, MulDF(df, x, Create2(df, Set(df, 0.868588984), Set(df, -2.170757285e-08))));
  s = AddFastDF(df, s, Mul(Mul(x2, Get2<0>(x)), t));

  Vec<D> r = Add(Get2<0>(s), Get2<1>(s));

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  r = IfThenElse(Eq(d, Inf(df)), Set(df, InfDouble), r);
  r = IfThenElse(Or(Lt(d, Set(df, 0)), IsNaN(d)), Set(df, NanDouble), r);
  r = IfThenElse(Eq(d, Set(df, 0)), Set(df, -InfDouble), r);
#else
  r = Fixup<0>(r, d, Set(di, (4 << (2*4)) | (3 << (4*4)) | (5 << (5*4)) | (2 << (6*4))));
#endif
  
  return r;
}

// Computes log2(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2757 xlog2f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Log2(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec2<D> x;
  Vec<D> t, m, x2;

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Mask<D> o = Lt(d, Set(df, FloatMin));
  d = IfThenElse(o, Mul(d, Set(df, (float)(INT64_C(1) << 32) * (float)(INT64_C(1) << 32))), d);
  Vec<RebindToSigned<D>> e = ILogB2(df, Mul(d, Set(df, 1.0/0.75)));
  m = LoadExp3(df, d, Neg(e));
  e = IfThenElse(RebindMask(di, o), Sub(e, Set(di, 64)), e);
#else
  Vec<D> e = GetExponent(Mul(d, Set(df, 1.0/0.75)));
  e = IfThenElse(Eq(e, Inf(df)), Set(df, 128.0f), e);
  m = GetMantissa(d);
#endif

  x = DivDF(df, AddDF(df, Set(df, -1), m), AddDF(df, Set(df, 1), m));
  x2 = Mul(Get2<0>(x), Get2<0>(x));

  t = Set(df, +0.4374550283e+0f);
  t = MulAdd(t, x2, Set(df, +0.5764790177e+0f));
  t = MulAdd(t, x2, Set(df, +0.9618012905120f));
  
#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Vec2<D> s = AddDF(df, ConvertTo(df, e),
				MulDF(df, x, Create2(df, Set(df, 2.8853900432586669922), Set(df, 3.2734474483568488616e-08))));
#else
  Vec2<D> s = AddDF(df, e,
				MulDF(df, x, Create2(df, Set(df, 2.8853900432586669922), Set(df, 3.2734474483568488616e-08))));
#endif

  s = AddDF(df, s, Mul(Mul(x2, Get2<0>(x)), t));

  Vec<D> r = Add(Get2<0>(s), Get2<1>(s));

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  r = IfThenElse(Eq(d, Inf(df)), Set(df, InfDouble), r);
  r = IfThenElse(Or(Lt(d, Set(df, 0)), IsNaN(d)), Set(df, NanDouble), r);
  r = IfThenElse(Eq(d, Set(df, 0)), Set(df, -InfDouble), r);
#else
  r = Fixup<0>(r, d, Set(di, (4 << (2*4)) | (3 << (4*4)) | (5 << (5*4)) | (2 << (6*4))));
#endif
  
  return r;
}

// Computes log1p(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2842 xlog1pf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Log1p(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec2<D> x;
  Vec<D> t, m, x2;

  Vec<D> dp1 = Add(d, Set(df, 1));

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Mask<D> o = Lt(dp1, Set(df, FloatMin));
  dp1 = IfThenElse(o, Mul(dp1, Set(df, (float)(INT64_C(1) << 32) * (float)(INT64_C(1) << 32))), dp1);
  Vec<RebindToSigned<D>> e = ILogB2(df, Mul(dp1, Set(df, 1.0f/0.75f)));
  t = LoadExp3(df, Set(df, 1), Neg(e));
  m = MulAdd(d, t, Sub(t, Set(df, 1)));
  e = IfThenElse(RebindMask(di, o), Sub(e, Set(di, 64)), e);
  Vec2<D> s = MulDF(df, Create2(df, Set(df, 0.69314718246459960938f), Set(df, -1.904654323148236017e-09f)), ConvertTo(df, e));
#else
  Vec<D> e = GetExponent(Mul(dp1, Set(df, 1.0f/0.75f)));
  e = IfThenElse(Eq(e, Inf(df)), Set(df, 128.0f), e);
  t = LoadExp3(df, Set(df, 1), Neg(NearestInt(e)));
  m = MulAdd(d, t, Sub(t, Set(df, 1)));
  Vec2<D> s = MulDF(df, Create2(df, Set(df, 0.69314718246459960938f), Set(df, -1.904654323148236017e-09f)), e);
#endif

  x = DivDF(df, Create2(df, m, Set(df, 0)), AddFastDF(df, Set(df, 2), m));
  x2 = Mul(Get2<0>(x), Get2<0>(x));

  t = Set(df, +0.3027294874e+0f);
  t = MulAdd(t, x2, Set(df, +0.3996108174e+0f));
  t = MulAdd(t, x2, Set(df, +0.6666694880e+0f));
  
  s = AddFastDF(df, s, ScaleDF(df, x, Set(df, 2)));
  s = AddFastDF(df, s, Mul(Mul(x2, Get2<0>(x)), t));

  Vec<D> r = Add(Get2<0>(s), Get2<1>(s));
  
  r = IfThenElse(Gt(d, Set(df, 1e+38)), Set(df, InfFloat), r);
  r = BitCast(df, IfThenElse(RebindMask(du, Gt(Set(df, -1), d)), Set(du, -1), BitCast(du, r)));
  r = IfThenElse(Eq(d, Set(df, -1)), Set(df, -InfFloat), r);
  r = IfThenElse(Eq(d, Set(df, -0.0)), Set(df, -0.0f), r);

  return r;
}

// Computes sqrt(x) with 0.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:2992 xsqrtf_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Sqrt(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
#if defined(ENABLE_FMA_SP)
  Vec<D> q, w, x, y, z;

  d = IfThenElse(Lt(d, Set(df, 0)), Set(df, NanFloat), d);

  Mask<D> o = Lt(d, Set(df, 5.2939559203393770e-23f));
  d = IfThenElse(o, Mul(d, Set(df, 1.8889465931478580e+22f)), d);
  q = IfThenElse(o, Set(df, 7.2759576141834260e-12f), Set(df, 1.0f));

  y = BitCast(df, Sub(Set(di, 0x5f3759df), BitCast(di, ShiftRight<1>(BitCast(du, BitCast(di, d))))));

  x = Mul(d, y);         w = Mul(Set(df, 0.5), y);
  y = NegMulAdd(x, w, Set(df, 0.5));
  x = MulAdd(x, y, x);   w = MulAdd(w, y, w);
  y = NegMulAdd(x, w, Set(df, 0.5));
  x = MulAdd(x, y, x);   w = MulAdd(w, y, w);

  y = NegMulAdd(x, w, Set(df, 1.5));  w = Add(w, w);
  w = Mul(w, y);
  x = Mul(w, d);
  y = MulSub(w, d, x); z = NegMulAdd(w, x, Set(df, 1));

  z = NegMulAdd(w, y, z); w = Mul(Set(df, 0.5), x);
  w = MulAdd(w, z, y);
  w = Add(w, x);

  w = Mul(w, q);

  w = IfThenElse(Or(Eq(d, Set(df, 0)), Eq(d, Set(df, InfFloat))), d, w);

  w = IfThenElse(Lt(d, Set(df, 0)), Set(df, NanFloat), w);

  return w;
#else
  Vec<D> q;
  Mask<D> o;
  
  d = IfThenElse(Lt(d, Set(df, 0)), Set(df, NanFloat), d);

  o = Lt(d, Set(df, 5.2939559203393770e-23f));
  d = IfThenElse(o, Mul(d, Set(df, 1.8889465931478580e+22f)), d);
  q = IfThenElse(o, Set(df, 7.2759576141834260e-12f*0.5f), Set(df, 0.5f));

  o = Gt(d, Set(df, 1.8446744073709552e+19f));
  d = IfThenElse(o, Mul(d, Set(df, 5.4210108624275220e-20f)), d);
  q = IfThenElse(o, Set(df, 4294967296.0f * 0.5f), q);

  Vec<D> x = BitCast(df, Sub(Set(di, 0x5f375a86), BitCast(di, ShiftRight<1>(BitCast(du, BitCast(di, Add(d, Set(df, 1e-45f))))))));

  x = Mul(x, Sub(Set(df, 1.5f), Mul(Mul(Mul(Set(df, 0.5f), d), x), x)));
  x = Mul(x, Sub(Set(df, 1.5f), Mul(Mul(Mul(Set(df, 0.5f), d), x), x)));
  x = Mul(x, Sub(Set(df, 1.5f), Mul(Mul(Mul(Set(df, 0.5f), d), x), x)));
  x = Mul(x, d);

  Vec2<D> d2 = MulDF(df, AddDF(df, d, MulDF(df, x, x)), RecDF(df, x));

  x = Mul(Add(Get2<0>(d2), Get2<1>(d2)), q);

  x = IfThenElse(Eq(d, Inf(df)), Set(df, InfFloat), x);
  x = IfThenElse(Eq(d, Set(df, 0)), d, x);
  
  return x;
#endif
}

// Computes sqrt(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:2073 xsqrtf_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) SqrtFast(const D df, Vec<D> d) {
#if HWY_ARCH_ARM && HWY_TARGET >= HWY_NEON
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> e = BitCast(df, Add(Set(di, 0x20000000), And(Set(di, 0x7f000000), BitCast(di, ShiftRight<1>(BitCast(du, BitCast(di, d)))))));
  Vec<D> m = BitCast(df, Add(Set(di, 0x3f000000), And(Set(di, 0x01ffffff), BitCast(di, d))));
  Vec<D> x = Vec<D>{vrsqrteq_f32(m.raw)};
  x = Mul(x, Vec<D>{vrsqrtsq_f32(m.raw, Mul(x, x).raw)});
  Vec<D> u = Mul(x, m);
  u = MulAdd(NegMulAdd(u, u, m), Mul(x, Set(df, 0.5)), u);
  e = BitCast(df, IfThenZeroElse(RebindMask(du, Eq(d, Set(df, 0))), BitCast(du, e)));
  u = Mul(e, u);

  u = IfThenElse(IsInf(d), Set(df, InfFloat), u);
  u = BitCast(df, IfThenElse(RebindMask(du, Or(IsNaN(d), Lt(d, Set(df, 0)))), Set(du, -1), BitCast(du, u)));
  u = MulSignBit(df, u, d);

  return u;
#else
 return Sqrt(d); 
#endif
}

// Computes cube root of x with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:2100 xcbrtf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) CbrtFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec<D> x, y, q = Set(df, 1.0), t;
  Vec<RebindToSigned<D>> e, qu, re;

#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
  Vec<D> s = d;
#endif
  e = Add(ILogB(df, Abs(d)), Set(di, 1));
  d = LoadExp2(df, d, Neg(e));

  t = Add(ConvertTo(df, e), Set(df, 6144));
  qu = ConvertTo(di, Mul(t, Set(df, 1.0f/3.0f)));
  re = ConvertTo(di, Sub(t, Mul(ConvertTo(df, qu), Set(df, 3))));

  q = IfThenElse(RebindMask(df, Eq(re, Set(di, 1))), Set(df, 1.2599210498948731647672106f), q);
  q = IfThenElse(RebindMask(df, Eq(re, Set(di, 2))), Set(df, 1.5874010519681994747517056f), q);
  q = LoadExp2(df, q, Sub(qu, Set(di, 2048)));

  q = MulSignBit(df, q, d);
  d = Abs(d);

  x = Set(df, -0.601564466953277587890625f);
  x = MulAdd(x, d, Set(df, 2.8208892345428466796875f));
  x = MulAdd(x, d, Set(df, -5.532182216644287109375f));
  x = MulAdd(x, d, Set(df, 5.898262500762939453125f));
  x = MulAdd(x, d, Set(df, -3.8095417022705078125f));
  x = MulAdd(x, d, Set(df, 2.2241256237030029296875f));

  y = Mul(Mul(d, x), x);
  y = Mul(Sub(y, Mul(Mul(Set(df, 2.0f / 3.0f), y), MulAdd(y, x, Set(df, -1.0f)))), q);

#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
  y = IfThenElse(IsInf(s), MulSignBit(df, Set(df, InfFloat), s), y);
  y = IfThenElse(Eq(s, Set(df, 0)), MulSignBit(df, Set(df, 0), s), y);
#endif
  
  return y;
}

// Computes cube root of x with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2141 xcbrtf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Cbrt(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec<D> x, y, z, t;
  Vec2<D> q2 = Create2(df, Set(df, 1), Set(df, 0)), u, v;
  Vec<RebindToSigned<D>> e, qu, re;

#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
  Vec<D> s = d;
#endif
  e = Add(ILogB(df, Abs(d)), Set(di, 1));
  d = LoadExp2(df, d, Neg(e));

  t = Add(ConvertTo(df, e), Set(df, 6144));
  qu = ConvertTo(di, Mul(t, Set(df, 1.0/3.0)));
  re = ConvertTo(di, Sub(t, Mul(ConvertTo(df, qu), Set(df, 3))));

  q2 = IfThenElse(df, RebindMask(df, Eq(re, Set(di, 1))), Create2(df, Set(df, 1.2599210739135742188f), Set(df, -2.4018701694217270415e-08)), q2);
  q2 = IfThenElse(df, RebindMask(df, Eq(re, Set(di, 2))), Create2(df, Set(df, 1.5874010324478149414f), Set(df, 1.9520385308169352356e-08)), q2);

  q2 = Set2<0>(q2, MulSignBit(df, Get2<0>(q2), d));
  q2 = Set2<1>(q2, MulSignBit(df, Get2<1>(q2), d));
  d = Abs(d);

  x = Set(df, -0.601564466953277587890625f);
  x = MulAdd(x, d, Set(df, 2.8208892345428466796875f));
  x = MulAdd(x, d, Set(df, -5.532182216644287109375f));
  x = MulAdd(x, d, Set(df, 5.898262500762939453125f));
  x = MulAdd(x, d, Set(df, -3.8095417022705078125f));
  x = MulAdd(x, d, Set(df, 2.2241256237030029296875f));

  y = Mul(x, x); y = Mul(y, y); x = Sub(x, Mul(NegMulAdd(d, y, x), Set(df, -1.0 / 3.0)));

  z = x;

  u = MulDF(df, x, x);
  u = MulDF(df, u, u);
  u = MulDF(df, u, d);
  u = AddDF(df, u, Neg(x));
  y = Add(Get2<0>(u), Get2<1>(u));

  y = Mul(Mul(Set(df, -2.0 / 3.0), y), z);
  v = AddDF(df, MulDF(df, z, z), y);
  v = MulDF(df, v, d);
  v = MulDF(df, v, q2);
  z = LoadExp2(df, Add(Get2<0>(v), Get2<1>(v)), Sub(qu, Set(di, 2048)));

  z = IfThenElse(IsInf(d), MulSignBit(df, Set(df, InfFloat), Get2<0>(q2)), z);
  z = IfThenElse(Eq(d, Set(df, 0)), BitCast(df, SignBit(df, Get2<0>(q2))), z);

#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
  z = IfThenElse(IsInf(s), MulSignBit(df, Set(df, InfFloat), s), z);
  z = IfThenElse(Eq(s, Set(df, 0)), MulSignBit(df, Set(df, 0), s), z);
#endif

  return z;
}

// Computes sqrt(x^2 + y^2) with 0.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:3069 xhypotf_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Hypot(const D df, Vec<D> x, Vec<D> y) {
  x = Abs(x);
  y = Abs(y);
  Vec<D> min = Min(x, y), n = min;
  Vec<D> max = Max(x, y), d = max;

  Mask<D> o = Lt(max, Set(df, FloatMin));
  n = IfThenElse(o, Mul(n, Set(df, UINT64_C(1) << 24)), n);
  d = IfThenElse(o, Mul(d, Set(df, UINT64_C(1) << 24)), d);

  Vec2<D> t = DivDF(df, Create2(df, n, Set(df, 0)), Create2(df, d, Set(df, 0)));
  t = MulDF(df, SqrtDF(df, AddDF(df, SquareDF(df, t), Set(df, 1))), max);
  Vec<D> ret = Add(Get2<0>(t), Get2<1>(t));
  ret = IfThenElse(IsNaN(ret), Set(df, InfFloat), ret);
  ret = IfThenElse(Eq(min, Set(df, 0)), max, ret);
  ret = IfThenElse(Or(IsNaN(x), IsNaN(y)), Set(df, NanFloat), ret);
  ret = IfThenElse(Or(Eq(x, Set(df, InfFloat)), Eq(y, Set(df, InfFloat))), Set(df, InfFloat), ret);

  return ret;
}

// Computes sqrt(x^2 + y^2) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:3090 xhypotf_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) HypotFast(const D df, Vec<D> x, Vec<D> y) {
  x = Abs(x);
  y = Abs(y);
  Vec<D> min = Min(x, y);
  Vec<D> max = Max(x, y);

  Vec<D> t = Div(min, max);
  Vec<D> ret = Mul(max, Sqrt(MulAdd(t, t, Set(df, 1))));
  ret = IfThenElse(Eq(min, Set(df, 0)), max, ret);
  ret = IfThenElse(Or(IsNaN(x), IsNaN(y)), Set(df, NanFloat), ret);
  ret = IfThenElse(Or(Eq(x, Set(df, InfFloat)), Eq(y, Set(df, InfFloat))), Set(df, InfFloat), ret);

  return ret;
}

// Computes x^y with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2360 xpowf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Pow(const D df, Vec<D> x, Vec<D> y) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
Mask<D> yisint = Or(Eq(Trunc(y), y), Gt(Abs(y), Set(df, 1 << 24)));
  Mask<D> yisodd = And(And(RebindMask(df, Eq(And(ConvertTo(di, y), Set(di, 1)), Set(di, 1))), yisint), Lt(Abs(y), Set(df, 1 << 24)));

#if HWY_ARCH_ARM && HWY_TARGET >= HWY_NEON
  yisodd = IfThenZeroElse(RebindMask(du, IsInf(y)), yisodd);
#endif

  Vec<D> result = ExpDF_float(df, MulDF(df, LogDF(df, Abs(x)), y));

  result = IfThenElse(IsNaN(result), Set(df, InfFloat), result);
  
  result = Mul(result, IfThenElse(Gt(x, Set(df, 0)), Set(df, 1), IfThenElse(yisint, IfThenElse(yisodd, Set(df, -1.0f), Set(df, 1)), Set(df, NanFloat))));

  Vec<D> efx = MulSignBit(df, Sub(Abs(x), Set(df, 1)), y);

  result = IfThenElse(IsInf(y), BitCast(df, IfThenZeroElse(RebindMask(du, Lt(efx, Set(df, 0.0f))), BitCast(du, IfThenElse(Eq(efx, Set(df, 0.0f)), Set(df, 1.0f), Set(df, InfFloat))))), result);

  result = IfThenElse(Or(IsInf(x), Eq(x, Set(df, 0.0))), MulSignBit(df, IfThenElse(Xor(SignBitMask(df, y), Eq(x, Set(df, 0.0f))), Set(df, 0), Set(df, InfFloat)),
					      IfThenElse(yisodd, x, Set(df, 1))), result);

  result = BitCast(df, IfThenElse(RebindMask(du, Or(IsNaN(x), IsNaN(y))), Set(du, -1), BitCast(du, result)));

  result = IfThenElse(Or(Eq(y, Set(df, 0)), Eq(x, Set(df, 1))), Set(df, 1), result);

  return result;

}

// Computes sin(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:969 xsinf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Sin(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<RebindToSigned<D>> q;
  Vec<D> u, v;
  Vec2<D> s, t, x;

  u = Round(Mul(d, Set(df, OneOverPi)));
  q = NearestInt(u);
  v = MulAdd(u, Set(df, -PiA2f), d);
  s = AddDF(df, v, Mul(u, Set(df, -PiB2f)));
  s = AddFastDF(df, s, Mul(u, Set(df, -PiC2f)));
  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2f));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    Vec3<D> dfi = PayneHanekReduction(df, d);
    Vec<RebindToSigned<D>> q2 = And(BitCast(di, Get3<2>(dfi)), Set(di, 3));
    q2 = Add(Add(q2, q2), IfThenElse(RebindMask(di, Gt(Get2<0>(Create2(df, Get3<0>(dfi), Get3<1>(dfi))), Set(df, 0))), Set(di, 2), Set(di, 1)));
    q2 = ShiftRight<2>(q2);
    Mask<D> o = RebindMask(df, Eq(And(BitCast(di, Get3<2>(dfi)), Set(di, 1)), Set(di, 1)));
    Vec2<D> x = Create2(df, MulSignBit(df, Set(df, 3.1415927410125732422f*-0.5), Get2<0>(Create2(df, Get3<0>(dfi), Get3<1>(dfi)))), MulSignBit(df, Set(df, -8.7422776573475857731e-08f*-0.5), Get2<0>(Create2(df, Get3<0>(dfi), Get3<1>(dfi)))));
    x = AddDF(df, Create2(df, Get3<0>(dfi), Get3<1>(dfi)), x);
    dfi = Set3<0>(Set3<1>(dfi, Get2<1>(IfThenElse(df, o, x, Create2(df, Get3<0>(dfi), Get3<1>(dfi))))), Get2<0>(IfThenElse(df, o, x, Create2(df, Get3<0>(dfi), Get3<1>(dfi)))));
    t = NormalizeDF(df, Create2(df, Get3<0>(dfi), Get3<1>(dfi)));

    t = Set2<0>(t, BitCast(df, IfThenElse(RebindMask(du, Or(IsInf(d), IsNaN(d))), Set(du, -1), BitCast(du, Get2<0>(t)))));

    q = IfThenElse(RebindMask(di, g), q, q2);
    s = IfThenElse(df, g, s, t);
  }

  t = s;
  s = SquareDF(df, s);

  u = Set(df, 2.6083159809786593541503e-06f);
  u = MulAdd(u, Get2<0>(s), Set(df, -0.0001981069071916863322258f));
  u = MulAdd(u, Get2<0>(s), Set(df, 0.00833307858556509017944336f));

  x = AddFastDF(df, Set(df, 1), MulDF(df, AddFastDF(df, Set(df, -0.166666597127914428710938f), Mul(u, Get2<0>(s))), s));

  u = MulDF_float(df, t, x);

  u = BitCast(df, Xor(IfThenElseZero(RebindMask(du, RebindMask(df, Eq(And(q, Set(di, 1)), Set(di, 1)))), BitCast(du, Set(df, -0.0))), BitCast(du, u)));

  u = IfThenElse(Eq(d, Set(df, -0.0)), d, u);

  return u; // #if !defined(DETERMINISTIC)
}

// Computes cos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1067 xcosf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Cos(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<RebindToSigned<D>> q;
  Vec<D> u;
  Vec2<D> s, t, x;

  Vec<D> dq = MulAdd(Round(MulAdd(d, Set(df, OneOverPi), Set(df, -0.5f))), Set(df, 2), Set(df, 1));
  q = NearestInt(dq);
  s = AddDF(df, d, Mul(dq, Set(df, -PiA2f*0.5f)));
  s = AddDF(df, s, Mul(dq, Set(df, -PiB2f*0.5f)));
  s = AddDF(df, s, Mul(dq, Set(df, -PiC2f*0.5f)));
  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2f));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    Vec3<D> dfi = PayneHanekReduction(df, d);
    Vec<RebindToSigned<D>> q2 = And(BitCast(di, Get3<2>(dfi)), Set(di, 3));
    q2 = Add(Add(q2, q2), IfThenElse(RebindMask(di, Gt(Get2<0>(Create2(df, Get3<0>(dfi), Get3<1>(dfi))), Set(df, 0))), Set(di, 8), Set(di, 7)));
    q2 = ShiftRight<1>(q2);
    Mask<D> o = RebindMask(df, Eq(And(BitCast(di, Get3<2>(dfi)), Set(di, 1)), Set(di, 0)));
    Vec<D> y = IfThenElse(Gt(Get2<0>(Create2(df, Get3<0>(dfi), Get3<1>(dfi))), Set(df, 0)), Set(df, 0), Set(df, -1));
    Vec2<D> x = Create2(df, MulSignBit(df, Set(df, 3.1415927410125732422f*-0.5), y), MulSignBit(df, Set(df, -8.7422776573475857731e-08f*-0.5), y));
    x = AddDF(df, Create2(df, Get3<0>(dfi), Get3<1>(dfi)), x);
    dfi = Set3<0>(Set3<1>(dfi, Get2<1>(IfThenElse(df, o, x, Create2(df, Get3<0>(dfi), Get3<1>(dfi))))), Get2<0>(IfThenElse(df, o, x, Create2(df, Get3<0>(dfi), Get3<1>(dfi)))));
    t = NormalizeDF(df, Create2(df, Get3<0>(dfi), Get3<1>(dfi)));

    t = Set2<0>(t, BitCast(df, IfThenElse(RebindMask(du, Or(IsInf(d), IsNaN(d))), Set(du, -1), BitCast(du, Get2<0>(t)))));

    q = IfThenElse(RebindMask(di, g), q, q2);
    s = IfThenElse(df, g, s, t);
  }

  t = s;
  s = SquareDF(df, s);

  u = Set(df, 2.6083159809786593541503e-06f);
  u = MulAdd(u, Get2<0>(s), Set(df, -0.0001981069071916863322258f));
  u = MulAdd(u, Get2<0>(s), Set(df, 0.00833307858556509017944336f));

  x = AddFastDF(df, Set(df, 1), MulDF(df, AddFastDF(df, Set(df, -0.166666597127914428710938f), Mul(u, Get2<0>(s))), s));

  u = MulDF_float(df, t, x);

  u = BitCast(df, Xor(IfThenElseZero(RebindMask(du, RebindMask(df, Eq(And(q, Set(di, 2)), Set(di, 0)))), BitCast(du, Set(df, -0.0))), BitCast(du, u)));
  
  return u; // #if !defined(DETERMINISTIC)
}

// Computes tan(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1635 xtanf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Tan(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<RebindToSigned<D>> q;
  Vec<D> u, v;
  Vec2<D> s, t, x;
  Mask<D> o;

  u = Round(Mul(d, Set(df, 2 * OneOverPi)));
  q = NearestInt(u);
  v = MulAdd(u, Set(df, -PiA2f*0.5f), d);
  s = AddDF(df, v, Mul(u, Set(df, -PiB2f*0.5f)));
  s = AddFastDF(df, s, Mul(u, Set(df, -PiC2f*0.5f)));
  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2f));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    Vec3<D> dfi = PayneHanekReduction(df, d);
    t = Create2(df, Get3<0>(dfi), Get3<1>(dfi));
    o = Or(IsInf(d), IsNaN(d));
    t = Set2<0>(t, BitCast(df, IfThenElse(RebindMask(du, o), Set(du, -1), BitCast(du, Get2<0>(t)))));
    t = Set2<1>(t, BitCast(df, IfThenElse(RebindMask(du, o), Set(du, -1), BitCast(du, Get2<1>(t)))));
    q = IfThenElse(RebindMask(di, g), q, BitCast(di, Get3<2>(dfi)));
    s = IfThenElse(df, g, s, t);
  }

  o = RebindMask(df, Eq(And(q, Set(di, 1)), Set(di, 1)));
  Vec<RebindToUnsigned<D>> n = IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0)));
  s = Set2<0>(s, BitCast(df, Xor(BitCast(du, Get2<0>(s)), n)));
  s = Set2<1>(s, BitCast(df, Xor(BitCast(du, Get2<1>(s)), n)));

  t = s;
  s = SquareDF(df, s);
  s = NormalizeDF(df, s);

  u = Set(df, 0.00446636462584137916564941f);
  u = MulAdd(u, Get2<0>(s), Set(df, -8.3920182078145444393158e-05f));
  u = MulAdd(u, Get2<0>(s), Set(df, 0.0109639242291450500488281f));
  u = MulAdd(u, Get2<0>(s), Set(df, 0.0212360303848981857299805f));
  u = MulAdd(u, Get2<0>(s), Set(df, 0.0540687143802642822265625f));

  x = AddFastDF(df, Set(df, 0.133325666189193725585938f), Mul(u, Get2<0>(s)));
  x = AddFastDF(df, Set(df, 1), MulDF(df, AddFastDF(df, Set(df, 0.33333361148834228515625f), MulDF(df, s, x)), s));
  x = MulDF(df, t, x);

  x = IfThenElse(df, o, RecDF(df, x), x);

  u = Add(Get2<0>(x), Get2<1>(x));

  u = IfThenElse(Eq(d, Set(df, -0.0)), d, u);
  
  return u; // #if !defined(DETERMINISTIC)
}

// Computes sin(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:630 xsinf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) SinFast(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<RebindToSigned<D>> q;
  Vec<D> u, s, r = d;

  q = NearestInt(Mul(d, Set(df, (float)OneOverPi)));
  u = ConvertTo(df, q);
  d = MulAdd(u, Set(df, -PiA2f), d);
  d = MulAdd(u, Set(df, -PiB2f), d);
  d = MulAdd(u, Set(df, -PiC2f), d);
  Mask<D> g = Lt(Abs(r), Set(df, TrigRangeMax2f));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    s = ConvertTo(df, q);
    u = MulAdd(s, Set(df, -PiAf), r);
    u = MulAdd(s, Set(df, -PiBf), u);
    u = MulAdd(s, Set(df, -PiCf), u);
    u = MulAdd(s, Set(df, -PiDf), u);

    d = IfThenElse(g, d, u);
    g = Lt(Abs(r), Set(df, TrigRangeMaxf));

    if (!HWY_LIKELY(AllTrue(df, g))) {
      Vec3<D> dfi = PayneHanekReduction(df, r);
      Vec<RebindToSigned<D>> q2 = And(BitCast(di, Get3<2>(dfi)), Set(di, 3));
      q2 = Add(Add(q2, q2), IfThenElse(RebindMask(di, Gt(Get2<0>(Create2(df, Get3<0>(dfi), Get3<1>(dfi))), Set(df, 0))), Set(di, 2), Set(di, 1)));
      q2 = ShiftRight<2>(q2);
      Mask<D> o = RebindMask(df, Eq(And(BitCast(di, Get3<2>(dfi)), Set(di, 1)), Set(di, 1)));
      Vec2<D> x = Create2(df, MulSignBit(df, Set(df, 3.1415927410125732422f*-0.5), Get2<0>(Create2(df, Get3<0>(dfi), Get3<1>(dfi)))), MulSignBit(df, Set(df, -8.7422776573475857731e-08f*-0.5), Get2<0>(Create2(df, Get3<0>(dfi), Get3<1>(dfi)))));
      x = AddDF(df, Create2(df, Get3<0>(dfi), Get3<1>(dfi)), x);
      dfi = Set3<0>(Set3<1>(dfi, Get2<1>(IfThenElse(df, o, x, Create2(df, Get3<0>(dfi), Get3<1>(dfi))))), Get2<0>(IfThenElse(df, o, x, Create2(df, Get3<0>(dfi), Get3<1>(dfi)))));
      u = Add(Get2<0>(Create2(df, Get3<0>(dfi), Get3<1>(dfi))), Get2<1>(Create2(df, Get3<0>(dfi), Get3<1>(dfi))));

      u = BitCast(df, IfThenElse(RebindMask(du, Or(IsInf(r), IsNaN(r))), Set(du, -1), BitCast(du, u)));

      q = IfThenElse(RebindMask(di, g), q, q2);
      d = IfThenElse(g, d, u);
    }
  }

  s = Mul(d, d);

  d = BitCast(df, Xor(IfThenElseZero(RebindMask(du, RebindMask(df, Eq(And(q, Set(di, 1)), Set(di, 1)))), BitCast(du, Set(df, -0.0f))), BitCast(du, d)));

  u = Set(df, 2.6083159809786593541503e-06f);
  u = MulAdd(u, s, Set(df, -0.0001981069071916863322258f));
  u = MulAdd(u, s, Set(df, 0.00833307858556509017944336f));
  u = MulAdd(u, s, Set(df, -0.166666597127914428710938f));

  u = Add(Mul(s, Mul(u, d)), d);

  u = IfThenElse(Eq(r, Set(df, -0.0)), r, u);

  return u; // #if !defined(DETERMINISTIC)
}

// Computes cos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:736 xcosf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) CosFast(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<RebindToSigned<D>> q;
  Vec<D> u, s, r = d;

  q = NearestInt(Sub(Mul(d, Set(df, (float)OneOverPi)), Set(df, 0.5f)));
  q = Add(Add(q, q), Set(di, 1));
  u = ConvertTo(df, q);
  d = MulAdd(u, Set(df, -PiA2f*0.5f), d);
  d = MulAdd(u, Set(df, -PiB2f*0.5f), d);
  d = MulAdd(u, Set(df, -PiC2f*0.5f), d);
  Mask<D> g = Lt(Abs(r), Set(df, TrigRangeMax2f));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    s = ConvertTo(df, q);
    u = MulAdd(s, Set(df, -PiAf*0.5f), r);
    u = MulAdd(s, Set(df, -PiBf*0.5f), u);
    u = MulAdd(s, Set(df, -PiCf*0.5f), u);
    u = MulAdd(s, Set(df, -PiDf*0.5f), u);

    d = IfThenElse(g, d, u);
    g = Lt(Abs(r), Set(df, TrigRangeMaxf));

    if (!HWY_LIKELY(AllTrue(df, g))) {
      Vec3<D> dfi = PayneHanekReduction(df, r);
      Vec<RebindToSigned<D>> q2 = And(BitCast(di, Get3<2>(dfi)), Set(di, 3));
      q2 = Add(Add(q2, q2), IfThenElse(RebindMask(di, Gt(Get2<0>(Create2(df, Get3<0>(dfi), Get3<1>(dfi))), Set(df, 0))), Set(di, 8), Set(di, 7)));
      q2 = ShiftRight<1>(q2);
      Mask<D> o = RebindMask(df, Eq(And(BitCast(di, Get3<2>(dfi)), Set(di, 1)), Set(di, 0)));
      Vec<D> y = IfThenElse(Gt(Get2<0>(Create2(df, Get3<0>(dfi), Get3<1>(dfi))), Set(df, 0)), Set(df, 0), Set(df, -1));
      Vec2<D> x = Create2(df, MulSignBit(df, Set(df, 3.1415927410125732422f*-0.5), y), MulSignBit(df, Set(df, -8.7422776573475857731e-08f*-0.5), y));
      x = AddDF(df, Create2(df, Get3<0>(dfi), Get3<1>(dfi)), x);
      dfi = Set3<0>(Set3<1>(dfi, Get2<1>(IfThenElse(df, o, x, Create2(df, Get3<0>(dfi), Get3<1>(dfi))))), Get2<0>(IfThenElse(df, o, x, Create2(df, Get3<0>(dfi), Get3<1>(dfi)))));
      u = Add(Get2<0>(Create2(df, Get3<0>(dfi), Get3<1>(dfi))), Get2<1>(Create2(df, Get3<0>(dfi), Get3<1>(dfi))));

      u = BitCast(df, IfThenElse(RebindMask(du, Or(IsInf(r), IsNaN(r))), Set(du, -1), BitCast(du, u)));

      q = IfThenElse(RebindMask(di, g), q, q2);
      d = IfThenElse(g, d, u);
    }
  }

  s = Mul(d, d);

  d = BitCast(df, Xor(IfThenElseZero(RebindMask(du, RebindMask(df, Eq(And(q, Set(di, 2)), Set(di, 0)))), BitCast(du, Set(df, -0.0f))), BitCast(du, d)));

  u = Set(df, 2.6083159809786593541503e-06f);
  u = MulAdd(u, s, Set(df, -0.0001981069071916863322258f));
  u = MulAdd(u, s, Set(df, 0.00833307858556509017944336f));
  u = MulAdd(u, s, Set(df, -0.166666597127914428710938f));

  u = Add(Mul(s, Mul(u, d)), d);

  return u; // #if !defined(DETERMINISTIC)
}

// Computes tan(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:845 xtanf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) TanFast(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<RebindToSigned<D>> q;
  Mask<D> o;
  Vec<D> u, s, x;

  q = NearestInt(Mul(d, Set(df, (float)(2 * OneOverPi))));
  u = ConvertTo(df, q);
  x = MulAdd(u, Set(df, -PiA2f*0.5f), d);
  x = MulAdd(u, Set(df, -PiB2f*0.5f), x);
  x = MulAdd(u, Set(df, -PiC2f*0.5f), x);
  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2f*0.5f));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    Vec<RebindToSigned<D>> q2 = NearestInt(Mul(d, Set(df, (float)(2 * OneOverPi))));
    s = ConvertTo(df, q);
    u = MulAdd(s, Set(df, -PiAf*0.5f), d);
    u = MulAdd(s, Set(df, -PiBf*0.5f), u);
    u = MulAdd(s, Set(df, -PiCf*0.5f), u);
    u = MulAdd(s, Set(df, -PiDf*0.5f), u);

    q = IfThenElse(RebindMask(di, g), q, q2);
    x = IfThenElse(g, x, u);
    g = Lt(Abs(d), Set(df, TrigRangeMaxf));

    if (!HWY_LIKELY(AllTrue(df, g))) {
      Vec3<D> dfi = PayneHanekReduction(df, d);
      u = Add(Get2<0>(Create2(df, Get3<0>(dfi), Get3<1>(dfi))), Get2<1>(Create2(df, Get3<0>(dfi), Get3<1>(dfi))));
      u = BitCast(df, IfThenElse(RebindMask(du, Or(IsInf(d), IsNaN(d))), Set(du, -1), BitCast(du, u)));
      u = IfThenElse(Eq(d, Set(df, -0.0)), d, u);
      q = IfThenElse(RebindMask(di, g), q, BitCast(di, Get3<2>(dfi)));
      x = IfThenElse(g, x, u);
    }
  }

  s = Mul(x, x);

  o = RebindMask(df, Eq(And(q, Set(di, 1)), Set(di, 1)));
  x = BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0f))), BitCast(du, x)));

#if HWY_ARCH_ARM && HWY_TARGET >= HWY_NEON
  u = Set(df, 0.00927245803177356719970703f);
  u = MulAdd(u, s, Set(df, 0.00331984995864331722259521f));
  u = MulAdd(u, s, Set(df, 0.0242998078465461730957031f));
  u = MulAdd(u, s, Set(df, 0.0534495301544666290283203f));
  u = MulAdd(u, s, Set(df, 0.133383005857467651367188f));
  u = MulAdd(u, s, Set(df, 0.333331853151321411132812f));
#else
  Vec<D> s2 = Mul(s, s), s4 = Mul(s2, s2);
  u = Estrin(s, s2, s4, Set(df, 0.333331853151321411132812f), Set(df, 0.133383005857467651367188f), Set(df, 0.0534495301544666290283203f), Set(df, 0.0242998078465461730957031f), Set(df, 0.00331984995864331722259521f), Set(df, 0.00927245803177356719970703f));
#endif

  u = MulAdd(s, Mul(u, x), x);

  u = IfThenElse(o, Div(Set(df, 1.0), u), u);

  return u; // #if !defined(DETERMINISTIC)
}

// Computes sinh(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2447 xsinhf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Sinh(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> y = Abs(x);
  Vec2<D> d = ExpDF(df, Create2(df, y, Set(df, 0)));
  d = SubDF(df, d, RecDF(df, d));
  y = Mul(Add(Get2<0>(d), Get2<1>(d)), Set(df, 0.5));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, 89)), IsNaN(y)), Set(df, InfFloat), y);
  y = MulSignBit(df, y, x);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

// Computes cosh(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2461 xcoshf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Cosh(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> y = Abs(x);
  Vec2<D> d = ExpDF(df, Create2(df, y, Set(df, 0)));
  d = AddFastDF(df, d, RecDF(df, d));
  y = Mul(Add(Get2<0>(d), Get2<1>(d)), Set(df, 0.5));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, 89)), IsNaN(y)), Set(df, InfFloat), y);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

// Computes tanh(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:2474 xtanhf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Tanh(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> y = Abs(x);
  Vec2<D> d = ExpDF(df, Create2(df, y, Set(df, 0)));
  Vec2<D> e = RecDF(df, d);
  d = DivDF(df, AddFastDF(df, d, NegDF(df, e)), AddFastDF(df, d, e));
  y = Add(Get2<0>(d), Get2<1>(d));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, 8.664339742f)), IsNaN(y)), Set(df, 1.0f), y);
  y = MulSignBit(df, y, x);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

// Computes sinh(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:2489 xsinhf_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) SinhFast(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> e = Expm1Fast(df, Abs(x));
  Vec<D> y = Div(Add(e, Set(df, 2)), Add(e, Set(df, 1)));
  y = Mul(y, Mul(Set(df, 0.5f), e));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, 88)), IsNaN(y)), Set(df, InfFloat), y);
  y = MulSignBit(df, y, x);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

// Computes cosh(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:2502 xcoshf_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) CoshFast(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> e = sleef::Exp(df, Abs(x));
  Vec<D> y = MulAdd(Set(df, 0.5f), e, Div(Set(df, 0.5), e));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, 88)), IsNaN(y)), Set(df, InfFloat), y);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

// Computes tanh(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:2513 xtanhf_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) TanhFast(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> d = Expm1Fast(df, Mul(Set(df, 2), Abs(x)));
  Vec<D> y = Div(d, Add(Set(df, 2), d));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, 8.664339742f)), IsNaN(y)), Set(df, 1.0f), y);
  y = MulSignBit(df, y, x);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

// Computes acos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1948 xacosf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Acos(const D df, Vec<D> d) {
  Mask<D> o = Lt(Abs(d), Set(df, 0.5f));
  Vec<D> x2 = IfThenElse(o, Mul(d, d), Mul(Sub(Set(df, 1), Abs(d)), Set(df, 0.5f))), u;
  Vec2<D> x = IfThenElse(df, o, Create2(df, Abs(d), Set(df, 0)), SqrtDF(df, x2));
  x = IfThenElse(df, Eq(Abs(d), Set(df, 1.0f)), Create2(df, Set(df, 0), Set(df, 0)), x);

  u = Set(df, +0.4197454825e-1);
  u = MulAdd(u, x2, Set(df, +0.2424046025e-1));
  u = MulAdd(u, x2, Set(df, +0.4547423869e-1));
  u = MulAdd(u, x2, Set(df, +0.7495029271e-1));
  u = MulAdd(u, x2, Set(df, +0.1666677296e+0));
  u = Mul(u, Mul(x2, Get2<0>(x)));

  Vec2<D> y = SubDF(df, Create2(df, Set(df, 3.1415927410125732422f/2), Set(df, -8.7422776573475857731e-08f/2)),
				 AddFastDF(df, MulSignBit(df, Get2<0>(x), d), MulSignBit(df, u, d)));
  x = AddFastDF(df, x, u);

  y = IfThenElse(df, o, y, ScaleDF(df, x, Set(df, 2)));
  
  y = IfThenElse(df, AndNot(o, Lt(d, Set(df, 0))),
			  SubDF(df, Create2(df, Set(df, 3.1415927410125732422f), Set(df, -8.7422776573475857731e-08f)), y), y);

  return Add(Get2<0>(y), Get2<1>(y));
}

// Computes asin(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1928 xasinf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Asin(const D df, Vec<D> d) {
  Mask<D> o = Lt(Abs(d), Set(df, 0.5f));
  Vec<D> x2 = IfThenElse(o, Mul(d, d), Mul(Sub(Set(df, 1), Abs(d)), Set(df, 0.5f))), u;
  Vec2<D> x = IfThenElse(df, o, Create2(df, Abs(d), Set(df, 0)), SqrtDF(df, x2));
  x = IfThenElse(df, Eq(Abs(d), Set(df, 1.0f)), Create2(df, Set(df, 0), Set(df, 0)), x);

  u = Set(df, +0.4197454825e-1);
  u = MulAdd(u, x2, Set(df, +0.2424046025e-1));
  u = MulAdd(u, x2, Set(df, +0.4547423869e-1));
  u = MulAdd(u, x2, Set(df, +0.7495029271e-1));
  u = MulAdd(u, x2, Set(df, +0.1666677296e+0));
  u = Mul(u, Mul(x2, Get2<0>(x)));

  Vec2<D> y = SubDF(df, SubDF(df, Create2(df, Set(df, 3.1415927410125732422f/4), Set(df, -8.7422776573475857731e-08f/4)), x), u);
  
  Vec<D> r = IfThenElse(o, Add(u, Get2<0>(x)), Mul(Add(Get2<0>(y), Get2<1>(y)), Set(df, 2)));
  return MulSignBit(df, r, d);
}

// Computes asinh(x) with 1 ULP accuracy
// Translated from libm/sleefsimdsp.c:2554 xasinhf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Asinh(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> y = Abs(x);
  Mask<D> o = Gt(y, Set(df, 1));
  Vec2<D> d;
  
  d = IfThenElse(df, o, RecDF(df, x), Create2(df, y, Set(df, 0)));
  d = SqrtDF(df, AddDF(df, SquareDF(df, d), Set(df, 1)));
  d = IfThenElse(df, o, MulDF(df, d, y), d);

  d = LogFastDF(df, NormalizeDF(df, AddDF(df, d, x)));
  y = Add(Get2<0>(d), Get2<1>(d));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, SqrtFloatMax)), IsNaN(y)), MulSignBit(df, Set(df, InfFloat), x), y);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));
  y = IfThenElse(Eq(x, Set(df, -0.0)), Set(df, -0.0), y);

  return y;
}

// Computes acos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1847 xacosf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) AcosFast(const D df, Vec<D> d) {
  Mask<D> o = Lt(Abs(d), Set(df, 0.5f));
  Vec<D> x2 = IfThenElse(o, Mul(d, d), Mul(Sub(Set(df, 1), Abs(d)), Set(df, 0.5f))), u;
  Vec<D> x = IfThenElse(o, Abs(d), Sqrt(x2));
  x = IfThenElse(Eq(Abs(d), Set(df, 1.0f)), Set(df, 0), x);

  u = Set(df, +0.4197454825e-1);
  u = MulAdd(u, x2, Set(df, +0.2424046025e-1));
  u = MulAdd(u, x2, Set(df, +0.4547423869e-1));
  u = MulAdd(u, x2, Set(df, +0.7495029271e-1));
  u = MulAdd(u, x2, Set(df, +0.1666677296e+0));
  u = Mul(u, Mul(x2, x));

  Vec<D> y = Sub(Set(df, 3.1415926535897932f/2), Add(MulSignBit(df, x, d), MulSignBit(df, u, d)));
  x = Add(x, u);
  Vec<D> r = IfThenElse(o, y, Mul(x, Set(df, 2)));
  return IfThenElse(AndNot(o, Lt(d, Set(df, 0))), Get2<0>(AddFastDF(df, Create2(df, Set(df, 3.1415927410125732422f), Set(df, -8.7422776573475857731e-08f)),
							  Neg(r))), r);
}

// Computes asin(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1831 xasinf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) AsinFast(const D df, Vec<D> d) {
  Mask<D> o = Lt(Abs(d), Set(df, 0.5f));
  Vec<D> x2 = IfThenElse(o, Mul(d, d), Mul(Sub(Set(df, 1), Abs(d)), Set(df, 0.5f)));
  Vec<D> x = IfThenElse(o, Abs(d), Sqrt(x2)), u;

  u = Set(df, +0.4197454825e-1);
  u = MulAdd(u, x2, Set(df, +0.2424046025e-1));
  u = MulAdd(u, x2, Set(df, +0.4547423869e-1));
  u = MulAdd(u, x2, Set(df, +0.7495029271e-1));
  u = MulAdd(u, x2, Set(df, +0.1666677296e+0));
  u = MulAdd(u, Mul(x, x2), x);

  Vec<D> r = IfThenElse(o, u, MulAdd(u, Set(df, -2), Set(df, Pif/2)));
  return MulSignBit(df, r, d);
}

// Computes atan(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1743 xatanf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) AtanFast(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> s, t, u;
  Vec<RebindToSigned<D>> q;

  q = SignBitOrZero(df, d, Set(di, 2));
  s = Abs(d);

  q = IfThenElse(RebindMask(di, Lt(Set(df, 1.0f), s)), Add(q, Set(di, 1)), q);
  s = IfThenElse(Lt(Set(df, 1.0f), s), Div(Set(df, 1.0), s), s);

  t = Mul(s, s);

  Vec<D> t2 = Mul(t, t), t4 = Mul(t2, t2);
  u = Estrin(t, t2, t4, Set(df, -0.333331018686294555664062f), Set(df, 0.199926957488059997558594f), Set(df, -0.142027363181114196777344f), Set(df, 0.106347933411598205566406f), Set(df, -0.0748900920152664184570312f), Set(df, 0.0425049886107444763183594f), Set(df, -0.0159569028764963150024414f), Set(df, 0.00282363896258175373077393f));

  t = MulAdd(s, Mul(t, u), s);

  t = IfThenElse(RebindMask(df, Eq(And(q, Set(di, 1)), Set(di, 1))), Sub(Set(df, (float)(Pi/2)), t), t);

  t = BitCast(df, Xor(IfThenElseZero(RebindMask(du, RebindMask(df, Eq(And(q, Set(di, 2)), Set(di, 2)))), BitCast(du, Set(df, -0.0f))), BitCast(du, t)));

#if HWY_ARCH_ARM && HWY_TARGET >= HWY_NEON
  t = IfThenElse(IsInf(d), MulSignBit(df, Set(df, 1.5874010519681994747517056f), d), t);
#endif

  return t;
}

// Computes acosh(x) with 1 ULP accuracy
// Translated from libm/sleefsimdsp.c:2575 xacoshf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Acosh(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec2<D> d = LogFastDF(df, AddDF(df, MulDF(df, SqrtDF(df, AddDF(df, x, Set(df, 1))), SqrtDF(df, AddDF(df, x, Set(df, -1)))), x));
  Vec<D> y = Add(Get2<0>(d), Get2<1>(d));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, SqrtFloatMax)), IsNaN(y)), Set(df, InfFloat), y);

  y = BitCast(df, IfThenZeroElse(RebindMask(du, Eq(x, Set(df, 1.0f))), BitCast(du, y)));

  y = BitCast(df, IfThenElse(RebindMask(du, Lt(x, Set(df, 1.0f))), Set(du, -1), BitCast(du, y)));
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

// Computes atan(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1973 xatanf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Atan(const D df, Vec<D> d) {
  Vec2<D> d2 = ATan2DF(df, Create2(df, Abs(d), Set(df, 0)), Create2(df, Set(df, 1), Set(df, 0)));
  Vec<D> r = Add(Get2<0>(d2), Get2<1>(d2));
  r = IfThenElse(IsInf(d), Set(df, 1.570796326794896557998982), r);
  return MulSignBit(df, r, d);
}

// Computes atanh(x) with 1 ULP accuracy
// Translated from libm/sleefsimdsp.c:2591 xatanhf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Atanh(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> y = Abs(x);
  Vec2<D> d = LogFastDF(df, DivDF(df, AddDF(df, Set(df, 1), y), AddDF(df, Set(df, 1), Neg(y))));
  y = BitCast(df, IfThenElse(RebindMask(du, Gt(y, Set(df, 1.0))), Set(du, -1), BitCast(du, IfThenElse(Eq(y, Set(df, 1.0)), Set(df, InfFloat), Mul(Add(Get2<0>(d), Get2<1>(d)), Set(df, 0.5))))));

  y = BitCast(df, IfThenElse(RebindMask(du, Or(IsInf(x), IsNaN(y))), Set(du, -1), BitCast(du, y)));
  y = MulSignBit(df, y, x);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

// Computes e^x
// Translated from libm/sleefsimddp.c:2146 xexp
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Exp(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> u = Round(Mul(d, Set(df, OneOverLn2_d))), s;
  Vec<RebindToSigned<D>> q = ConvertTo(di, Round(u));

  s = MulAdd(u, Set(df, -Ln2Hi_d), d);
  s = MulAdd(u, Set(df, -Ln2Lo_d), s);

#ifdef HWY_SLEEF_HAS_FMA
  Vec<D> s2 = Mul(s, s), s4 = Mul(s2, s2), s8 = Mul(s4, s4);
  u = Estrin(s, s2, s4, s8, Set(df, +0.1666666666666669072e+0), Set(df, +0.4166666666666602598e-1), Set(df, +0.8333333333314938210e-2), Set(df, +0.1388888888914497797e-2), Set(df, +0.1984126989855865850e-3), Set(df, +0.2480158687479686264e-4), Set(df, +0.2755723402025388239e-5), Set(df, +0.2755762628169491192e-6), Set(df, +0.2511210703042288022e-7), Set(df, +0.2081276378237164457e-8));
  u = MulAdd(u, s, Set(df, +0.5000000000000000000e+0));
  u = MulAdd(u, s, Set(df, +0.1000000000000000000e+1));
  u = MulAdd(u, s, Set(df, +0.1000000000000000000e+1));
#else // #ifdef ENABLE_FMA_DP
  Vec<D> s2 = Mul(s, s), s4 = Mul(s2, s2), s8 = Mul(s4, s4);
  u = Estrin(s, s2, s4, s8, Set(df, 0.166666666666666851703837), Set(df, 0.0416666666666665047591422), Set(df, 0.00833333333331652721664984), Set(df, 0.00138888888889774492207962), Set(df, 0.000198412698960509205564975), Set(df, 2.4801587159235472998791e-05), Set(df, 2.75572362911928827629423e-06), Set(df, 2.75573911234900471893338e-07), Set(df, 2.51112930892876518610661e-08), Set(df, 2.08860621107283687536341e-09));
  u = MulAdd(u, s, Set(df, +0.5000000000000000000e+0));

  u = Add(Set(df, 1), MulAdd(Mul(s, s), u, s));
#endif // #ifdef ENABLE_FMA_DP
  
  u = LoadExp2(df, u, q);

  u = IfThenElse(Gt(d, Set(df, 709.78271114955742909217217426)), Set(df, InfDouble), u);
  u = BitCast(df, IfThenZeroElse(RebindMask(du, Lt(d, Set(df, -1000))), BitCast(du, u)));

  return u;
}

// Computes 2^x
// Translated from libm/sleefsimddp.c:2686 xexp2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Exp2(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> u = Round(d), s;
  Vec<RebindToSigned<D>> q = ConvertTo(di, Round(u));

  s = Sub(d, u);

  Vec<D> s2 = Mul(s, s), s4 = Mul(s2, s2), s8 = Mul(s4, s4);
  u = Estrin(s, s2, s4, s8, Set(df, +0.2402265069591012214e+0), Set(df, +0.5550410866482046596e-1), Set(df, +0.9618129107597600536e-2), Set(df, +0.1333355814670499073e-2), Set(df, +0.1540353045101147808e-3), Set(df, +0.1525273353517584730e-4), Set(df, +0.1321543872511327615e-5), Set(df, +0.1017819260921760451e-6), Set(df, +0.7073164598085707425e-8), Set(df, +0.4434359082926529454e-9));
  u = MulAdd(u, s, Set(df, +0.6931471805599452862e+0));
  
#ifdef HWY_SLEEF_HAS_FMA
  u = MulAdd(u, s, Set(df, 1));
#else
  u = Get2<0>(NormalizeDD(df, AddFastDD(df, Set(df, 1), MulDD(df, u, s))));
#endif
  
  u = LoadExp2(df, u, q);

  u = IfThenElse(Ge(d, Set(df, 1024)), Set(df, InfDouble), u);
  u = BitCast(df, IfThenZeroElse(RebindMask(du, Lt(d, Set(df, -2000))), BitCast(du, u)));

  return u;
}

// Computes 10^x
// Translated from libm/sleefsimddp.c:2750 xexp10
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Exp10(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> u = Round(Mul(d, Set(df, Lg10))), s;
  Vec<RebindToSigned<D>> q = ConvertTo(di, Round(u));

  s = MulAdd(u, Set(df, -Log10Of2_Hi_d), d);
  s = MulAdd(u, Set(df, -Log10Of2_Lo_d), s);

  u = Set(df, +0.2411463498334267652e-3);
  u = MulAdd(u, s, Set(df, +0.1157488415217187375e-2));
  u = MulAdd(u, s, Set(df, +0.5013975546789733659e-2));
  u = MulAdd(u, s, Set(df, +0.1959762320720533080e-1));
  u = MulAdd(u, s, Set(df, +0.6808936399446784138e-1));
  u = MulAdd(u, s, Set(df, +0.2069958494722676234e+0));
  u = MulAdd(u, s, Set(df, +0.5393829292058536229e+0));
  u = MulAdd(u, s, Set(df, +0.1171255148908541655e+1));
  u = MulAdd(u, s, Set(df, +0.2034678592293432953e+1));
  u = MulAdd(u, s, Set(df, +0.2650949055239205876e+1));
  u = MulAdd(u, s, Set(df, +0.2302585092994045901e+1));
  
#ifdef HWY_SLEEF_HAS_FMA
  u = MulAdd(u, s, Set(df, 1));
#else
  u = Get2<0>(NormalizeDD(df, AddFastDD(df, Set(df, 1), MulDD(df, u, s))));
#endif
  
  u = LoadExp2(df, u, q);

  u = IfThenElse(Gt(d, Set(df, 308.25471555991671)), Set(df, InfDouble), u);
  u = BitCast(df, IfThenZeroElse(RebindMask(du, Lt(d, Set(df, -350))), BitCast(du, u)));

  return u;
}

// Computes e^x - 1
// Translated from libm/sleefsimddp.c:2815 xexpm1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Expm1(const D df, Vec<D> a) {
  Vec2<D> d = AddDD(df, ExpDD(df, Create2(df, a, Set(df, 0))), Set(df, -1.0));
  Vec<D> x = Add(Get2<0>(d), Get2<1>(d));
  x = IfThenElse(Gt(a, Set(df, 709.782712893383996732223)), Set(df, InfDouble), x);
  x = IfThenElse(Lt(a, Set(df, -36.736800569677101399113302437)), Set(df, -1), x);
  x = IfThenElse(Eq(a, Set(df, -0.0)), Set(df, -0.0), x);
  return x;
}

// Computes ln(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2270 xlog_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Log(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec2<D> x;
  Vec<D> t, m, x2;

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Mask<D> o = Lt(d, Set(df, DoubleMin));
  d = IfThenElse(o, Mul(d, Set(df, (double)(INT64_C(1) << 32) * (double)(INT64_C(1) << 32))), d);
  Vec<RebindToSigned<D>> e = ILogB2(df, Mul(d, Set(df, 1.0/0.75)));
  m = LoadExp3(df, d, Neg(e));
  e = IfThenElse(RebindMask(di, o), Sub(e, Set(di, 64)), e);
#else
  Vec<D> e = GetExponent(Mul(d, Set(df, 1.0/0.75)));
  e = IfThenElse(Eq(e, Inf(df)), Set(df, 1024.0), e);
  m = GetMantissa(d);
#endif

  x = DivDD(df, AddDD(df, Set(df, -1), m), AddDD(df, Set(df, 1), m));
  x2 = Mul(Get2<0>(x), Get2<0>(x));

  Vec<D> x4 = Mul(x2, x2), x8 = Mul(x4, x4);
  t = Estrin(x2, x4, x8, Set(df, 0.6666666666667333541e+0), Set(df, 0.3999999999635251990e+0), Set(df, 0.2857142932794299317e+0), Set(df, 0.2222214519839380009e+0), Set(df, 0.1818605932937785996e+0), Set(df, 0.1525629051003428716e+0), Set(df, 0.1532076988502701353e+0));

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Vec2<D> s = MulDD(df, Create2(df, Set(df, 0.693147180559945286226764), Set(df, 2.319046813846299558417771e-17)), ConvertTo(df, e));
#else
  Vec2<D> s = MulDD(df, Create2(df, Set(df, 0.693147180559945286226764), Set(df, 2.319046813846299558417771e-17)), e);
#endif

  s = AddFastDD(df, s, ScaleDD(df, x, Set(df, 2)));
  s = AddFastDD(df, s, Mul(Mul(x2, Get2<0>(x)), t));

  Vec<D> r = Add(Get2<0>(s), Get2<1>(s));

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  r = IfThenElse(Eq(d, Inf(df)), Set(df, InfDouble), r);
  r = IfThenElse(Or(Lt(d, Set(df, 0)), IsNaN(d)), Set(df, NanDouble), r);
  r = IfThenElse(Eq(d, Set(df, 0)), Set(df, -InfDouble), r);
#else
  r = Fixup<0>(r, d, Set(di, (4 << (2*4)) | (3 << (4*4)) | (5 << (5*4)) | (2 << (6*4))));
#endif
  
  return r;
}

// Computes ln(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:2099 xlog
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) LogFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec<D> x, x2;
  Vec<D> t, m;
  
#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Mask<D> o = Lt(d, Set(df, DoubleMin));
  d = IfThenElse(o, Mul(d, Set(df, (double)(INT64_C(1) << 32) * (double)(INT64_C(1) << 32))), d);
  Vec<RebindToSigned<D>> e = ILogB2(df, Mul(d, Set(df, 1.0/0.75)));
  m = LoadExp3(df, d, Neg(e));
  e = IfThenElse(RebindMask(di, o), Sub(e, Set(di, 64)), e);
#else
  Vec<D> e = GetExponent(Mul(d, Set(df, 1.0/0.75)));
  e = IfThenElse(Eq(e, Inf(df)), Set(df, 1024.0), e);
  m = GetMantissa(d);
#endif
  
  x = Div(Sub(m, Set(df, 1)), Add(Set(df, 1), m));
  x2 = Mul(x, x);

  Vec<D> x4 = Mul(x2, x2), x8 = Mul(x4, x4), x3 = Mul(x, x2);
  t = Estrin(x2, x4, x8, Set(df, 0.6666666666667778740063), Set(df, 0.399999999950799600689777), Set(df, 0.285714294746548025383248), Set(df, 0.222221366518767365905163), Set(df, 0.181863266251982985677316), Set(df, 0.152519917006351951593857), Set(df, 0.153487338491425068243146));

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  x = MulAdd(x, Set(df, 2), Mul(Set(df, 0.693147180559945286226764), ConvertTo(df, e)));
  x = MulAdd(x3, t, x);

  x = IfThenElse(Eq(d, Inf(df)), Set(df, InfDouble), x);
  x = IfThenElse(Or(Lt(d, Set(df, 0)), IsNaN(d)), Set(df, NanDouble), x);
  x = IfThenElse(Eq(d, Set(df, 0)), Set(df, -InfDouble), x);
#else
  x = MulAdd(x, Set(df, 2), Mul(Set(df, 0.693147180559945286226764), e));
  x = MulAdd(x3, t, x);

  x = Fixup<0>(x, d, Set(di, (5 << (5*4))));
#endif

  return x;
}

// Computes log10(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2824 xlog10
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Log10(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec2<D> x;
  Vec<D> t, m, x2;

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Mask<D> o = Lt(d, Set(df, DoubleMin));
  d = IfThenElse(o, Mul(d, Set(df, (double)(INT64_C(1) << 32) * (double)(INT64_C(1) << 32))), d);
  Vec<RebindToSigned<D>> e = ILogB2(df, Mul(d, Set(df, 1.0/0.75)));
  m = LoadExp3(df, d, Neg(e));
  e = IfThenElse(RebindMask(di, o), Sub(e, Set(di, 64)), e);
#else
  Vec<D> e = GetExponent(Mul(d, Set(df, 1.0/0.75)));
  e = IfThenElse(Eq(e, Inf(df)), Set(df, 1024.0), e);
  m = GetMantissa(d);
#endif

  x = DivDD(df, AddDD(df, Set(df, -1), m), AddDD(df, Set(df, 1), m));
  x2 = Mul(Get2<0>(x), Get2<0>(x));

  Vec<D> x4 = Mul(x2, x2), x8 = Mul(x4, x4);
  t = Estrin(x2, x4, x8, Set(df, +0.2895296546021972617e+0), Set(df, +0.1737177927454605086e+0), Set(df, +0.1240841409721444993e+0), Set(df, +0.9650955035715275132e-1), Set(df, +0.7898105214313944078e-1), Set(df, +0.6625722782820833712e-1), Set(df, +0.6653725819576758460e-1));
  
#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Vec2<D> s = MulDD(df, Create2(df, Set(df, 0.30102999566398119802), Set(df, -2.803728127785170339e-18)), ConvertTo(df, e));
#else
  Vec2<D> s = MulDD(df, Create2(df, Set(df, 0.30102999566398119802), Set(df, -2.803728127785170339e-18)), e);
#endif

  s = AddFastDD(df, s, MulDD(df, x, Create2(df, Set(df, 0.86858896380650363334), Set(df, 1.1430059694096389311e-17))));
  s = AddFastDD(df, s, Mul(Mul(x2, Get2<0>(x)), t));

  Vec<D> r = Add(Get2<0>(s), Get2<1>(s));

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  r = IfThenElse(Eq(d, Inf(df)), Set(df, InfDouble), r);
  r = IfThenElse(Or(Lt(d, Set(df, 0)), IsNaN(d)), Set(df, NanDouble), r);
  r = IfThenElse(Eq(d, Set(df, 0)), Set(df, -InfDouble), r);
#else
  r = Fixup<0>(r, d, Set(di, (4 << (2*4)) | (3 << (4*4)) | (5 << (5*4)) | (2 << (6*4))));
#endif
  
  return r;
}

// Computes log2(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2875 xlog2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Log2(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec2<D> x;
  Vec<D> t, m, x2;

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Mask<D> o = Lt(d, Set(df, DoubleMin));
  d = IfThenElse(o, Mul(d, Set(df, (double)(INT64_C(1) << 32) * (double)(INT64_C(1) << 32))), d);
  Vec<RebindToSigned<D>> e = ILogB2(df, Mul(d, Set(df, 1.0/0.75)));
  m = LoadExp3(df, d, Neg(e));
  e = IfThenElse(RebindMask(di, o), Sub(e, Set(di, 64)), e);
#else
  Vec<D> e = GetExponent(Mul(d, Set(df, 1.0/0.75)));
  e = IfThenElse(Eq(e, Inf(df)), Set(df, 1024.0), e);
  m = GetMantissa(d);
#endif

  x = DivDD(df, AddDD(df, Set(df, -1), m), AddDD(df, Set(df, 1), m));
  x2 = Mul(Get2<0>(x), Get2<0>(x));

  Vec<D> x4 = Mul(x2, x2), x8 = Mul(x4, x4);
  t = Estrin(x2, x4, x8, Set(df, +0.96179669392608091449), Set(df, +0.5770780162997058982e+0), Set(df, +0.4121985945485324709e+0), Set(df, +0.3205977477944495502e+0), Set(df, +0.2623708057488514656e+0), Set(df, +0.2200768693152277689e+0), Set(df, +0.2211941750456081490e+0));
  
#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Vec2<D> s = AddDD(df, ConvertTo(df, e),
				 MulDD(df, x, Create2(df, Set(df, 2.885390081777926774), Set(df, 6.0561604995516736434e-18))));
#else
  Vec2<D> s = AddDD(df, e,
				 MulDD(df, x, Create2(df, Set(df, 2.885390081777926774), Set(df, 6.0561604995516736434e-18))));
#endif

  s = AddDD(df, s, Mul(Mul(x2, Get2<0>(x)), t));

  Vec<D> r = Add(Get2<0>(s), Get2<1>(s));

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  r = IfThenElse(Eq(d, Inf(df)), Set(df, InfDouble), r);
  r = IfThenElse(Or(Lt(d, Set(df, 0)), IsNaN(d)), Set(df, NanDouble), r);
  r = IfThenElse(Eq(d, Set(df, 0)), Set(df, -InfDouble), r);
#else
  r = Fixup<0>(r, d, Set(di, (4 << (2*4)) | (3 << (4*4)) | (5 << (5*4)) | (2 << (6*4))));
#endif
  
  return r;
}

// Computes log1p(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2974 xlog1p
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Log1p(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec2<D> x;
  Vec<D> t, m, x2;

  Vec<D> dp1 = Add(d, Set(df, 1));

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  Mask<D> o = Lt(dp1, Set(df, DoubleMin));
  dp1 = IfThenElse(o, Mul(dp1, Set(df, (double)(INT64_C(1) << 32) * (double)(INT64_C(1) << 32))), dp1);
  Vec<RebindToSigned<D>> e = ILogB2(df, Mul(dp1, Set(df, 1.0/0.75)));
  t = LoadExp3(df, Set(df, 1), Neg(e));
  m = MulAdd(d, t, Sub(t, Set(df, 1)));
  e = IfThenElse(RebindMask(di, o), Sub(e, Set(di, 64)), e);
  Vec2<D> s = MulDD(df, Create2(df, Set(df, 0.693147180559945286226764), Set(df, 2.319046813846299558417771e-17)), ConvertTo(df, e));
#else
  Vec<D> e = GetExponent(Mul(dp1, Set(df, 1.0/0.75)));
  e = IfThenElse(Eq(e, Inf(df)), Set(df, 1024.0), e);
  t = LoadExp3(df, Set(df, 1), Neg(ConvertTo(di, Round(e))));
  m = MulAdd(d, t, Sub(t, Set(df, 1)));
  Vec2<D> s = MulDD(df, Create2(df, Set(df, 0.693147180559945286226764), Set(df, 2.319046813846299558417771e-17)), e);
#endif

  x = DivDD(df, Create2(df, m, Set(df, 0)), AddFastDD(df, Set(df, 2), m));
  x2 = Mul(Get2<0>(x), Get2<0>(x));

  Vec<D> x4 = Mul(x2, x2), x8 = Mul(x4, x4);
  t = Estrin(x2, x4, x8, Set(df, 0.6666666666667333541e+0), Set(df, 0.3999999999635251990e+0), Set(df, 0.2857142932794299317e+0), Set(df, 0.2222214519839380009e+0), Set(df, 0.1818605932937785996e+0), Set(df, 0.1525629051003428716e+0), Set(df, 0.1532076988502701353e+0));
  
  s = AddFastDD(df, s, ScaleDD(df, x, Set(df, 2)));
  s = AddFastDD(df, s, Mul(Mul(x2, Get2<0>(x)), t));

  Vec<D> r = Add(Get2<0>(s), Get2<1>(s));
  
  r = IfThenElse(Gt(d, Set(df, 1e+307)), Set(df, InfDouble), r);
  r = IfThenElse(Or(Lt(d, Set(df, -1)), IsNaN(d)), Set(df, NanDouble), r);
  r = IfThenElse(Eq(d, Set(df, -1)), Set(df, -InfDouble), r);
  r = IfThenElse(Eq(d, Set(df, -0.0)), Set(df, -0.0), r);
  
  return r;
}

// Computes sqrt(x) with 0.5 ULP accuracy
// Translated from libm/sleefsimddp.c:3142 xsqrt_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Sqrt(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  
#if defined(ENABLE_FMA_DP)
  Vec<D> q, w, x, y, z;
  
  d = IfThenElse(Lt(d, Set(df, 0)), Set(df, NanDouble), d);

  Mask<D> o = Lt(d, Set(df, 8.636168555094445E-78));
  d = IfThenElse(o, Mul(d, Set(df, 1.157920892373162E77)), d);
  q = IfThenElse(o, Set(df, 2.9387358770557188E-39), Set(df, 1));

  y = BitCast(df, Sub(Set(du, (static_cast<uint64_t>(0x5fe6ec85) << 32) | 0xe7de30da), ShiftRight<1>(BitCast(du, d))));

  x = Mul(d, y);         w = Mul(Set(df, 0.5), y);
  y = NegMulAdd(x, w, Set(df, 0.5));
  x = MulAdd(x, y, x);   w = MulAdd(w, y, w);
  y = NegMulAdd(x, w, Set(df, 0.5));
  x = MulAdd(x, y, x);   w = MulAdd(w, y, w);
  y = NegMulAdd(x, w, Set(df, 0.5));
  x = MulAdd(x, y, x);   w = MulAdd(w, y, w);

  y = NegMulAdd(x, w, Set(df, 1.5));  w = Add(w, w);
  w = Mul(w, y);
  x = Mul(w, d);
  y = MulSub(w, d, x); z = NegMulAdd(w, x, Set(df, 1));

  z = NegMulAdd(w, y, z); w = Mul(Set(df, 0.5), x);
  w = MulAdd(w, z, y);
  w = Add(w, x);

  w = Mul(w, q);

  w = IfThenElse(Or(Eq(d, Set(df, 0)), Eq(d, Set(df, InfDouble))), d, w);

  w = IfThenElse(Lt(d, Set(df, 0)), Set(df, NanDouble), w);

  return w;
#else
  Vec<D> q;
  Mask<D> o;
  
  d = IfThenElse(Lt(d, Set(df, 0)), Set(df, NanDouble), d);

  o = Lt(d, Set(df, 8.636168555094445E-78));
  d = IfThenElse(o, Mul(d, Set(df, 1.157920892373162E77)), d);
  q = IfThenElse(o, Set(df, 2.9387358770557188E-39*0.5), Set(df, 0.5));

  o = Gt(d, Set(df, 1.3407807929942597e+154));
  d = IfThenElse(o, Mul(d, Set(df, 7.4583407312002070e-155)), d);
  q = IfThenElse(o, Set(df, 1.1579208923731620e+77*0.5), q);

  Vec<D> x = BitCast(df, Sub(Set(du, (static_cast<uint64_t>(0x5fe6ec86) << 32) | 0), ShiftRight<1>(BitCast(du, Add(d, Set(df, 1e-320))))));

  x = Mul(x, Sub(Set(df, 1.5), Mul(Mul(Mul(Set(df, 0.5), d), x), x)));
  x = Mul(x, Sub(Set(df, 1.5), Mul(Mul(Mul(Set(df, 0.5), d), x), x)));
  x = Mul(x, Sub(Set(df, 1.5), Mul(Mul(Mul(Set(df, 0.5), d), x), x)));
  x = Mul(x, d);

  Vec2<D> d2 = MulDD(df, AddDD(df, d, MulDD(df, x, x)), RecDD(df, x));

  x = Mul(Add(Get2<0>(d2), Get2<1>(d2)), q);

  x = IfThenElse(Eq(d, Inf(df)), Set(df, InfDouble), x);
  x = IfThenElse(Eq(d, Set(df, 0)), d, x);
  
  return x;
#endif
}

// Computes sqrt(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:3220 xsqrt_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) SqrtFast(const D df, Vec<D> d) {
 return sleef::Sqrt(df, d); }

// Computes cube root of x with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:2588 xcbrt
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) CbrtFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec<D> x, y, q = Set(df, 1.0);
  Vec<RebindToSigned<D>> e, qu, re;
  Vec<D> t;

#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
  Vec<D> s = d;
#endif
  e = Add(ILogB(df, Abs(d)), Set(di, 1));
  d = LoadExp2(df, d, Neg(e));

  t = Add(ConvertTo(df, e), Set(df, 6144));
  qu = ConvertTo(di, Trunc(Mul(t, Set(df, 1.0/3.0))));
  re = ConvertTo(di, Trunc(Sub(t, Mul(ConvertTo(df, qu), Set(df, 3)))));

  q = IfThenElse(RebindMask(df, Eq(re, Set(di, 1))), Set(df, 1.2599210498948731647672106), q);
  q = IfThenElse(RebindMask(df, Eq(re, Set(di, 2))), Set(df, 1.5874010519681994747517056), q);
  q = LoadExp2(df, q, Sub(qu, Set(di, 2048)));

  q = MulSignBit(df, q, d);

  d = Abs(d);

  x = Set(df, -0.640245898480692909870982);
  x = MulAdd(x, d, Set(df, 2.96155103020039511818595));
  x = MulAdd(x, d, Set(df, -5.73353060922947843636166));
  x = MulAdd(x, d, Set(df, 6.03990368989458747961407));
  x = MulAdd(x, d, Set(df, -3.85841935510444988821632));
  x = MulAdd(x, d, Set(df, 2.2307275302496609725722));

  y = Mul(x, x); y = Mul(y, y); x = Sub(x, Mul(MulSub(d, y, x), Set(df, 1.0 / 3.0)));
  y = Mul(Mul(d, x), x);
  y = Mul(Sub(y, Mul(Mul(Set(df, 2.0 / 3.0), y), MulAdd(y, x, Set(df, -1.0)))), q);

#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
  y = IfThenElse(IsInf(s), MulSignBit(df, Set(df, InfDouble), s), y);
  y = IfThenElse(Eq(s, Set(df, 0)), MulSignBit(df, Set(df, 0), s), y);
#endif
  
  return y;
}

// Computes cube root of x with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2630 xcbrt_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Cbrt(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec<D> x, y, z, t;
  Vec2<D> q2 = Create2(df, Set(df, 1), Set(df, 0)), u, v;
  Vec<RebindToSigned<D>> e, qu, re;

#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
  Vec<D> s = d;
#endif
  e = Add(ILogB(df, Abs(d)), Set(di, 1));
  d = LoadExp2(df, d, Neg(e));

  t = Add(ConvertTo(df, e), Set(df, 6144));
  qu = ConvertTo(di, Trunc(Mul(t, Set(df, 1.0/3.0))));
  re = ConvertTo(di, Trunc(Sub(t, Mul(ConvertTo(df, qu), Set(df, 3)))));

  q2 = IfThenElse(df, RebindMask(df, Eq(re, Set(di, 1))), Create2(df, Set(df, 1.2599210498948731907), Set(df, -2.5899333753005069177e-17)), q2);
  q2 = IfThenElse(df, RebindMask(df, Eq(re, Set(di, 2))), Create2(df, Set(df, 1.5874010519681995834), Set(df, -1.0869008194197822986e-16)), q2);

  q2 = Create2(df, MulSignBit(df, Get2<0>(q2), d), MulSignBit(df, Get2<1>(q2), d));
  d = Abs(d);

  x = Set(df, -0.640245898480692909870982);
  x = MulAdd(x, d, Set(df, 2.96155103020039511818595));
  x = MulAdd(x, d, Set(df, -5.73353060922947843636166));
  x = MulAdd(x, d, Set(df, 6.03990368989458747961407));
  x = MulAdd(x, d, Set(df, -3.85841935510444988821632));
  x = MulAdd(x, d, Set(df, 2.2307275302496609725722));

  y = Mul(x, x); y = Mul(y, y); x = Sub(x, Mul(MulSub(d, y, x), Set(df, 1.0 / 3.0)));

  z = x;

  u = MulDD(df, x, x);
  u = MulDD(df, u, u);
  u = MulDD(df, u, d);
  u = AddDD(df, u, Neg(x));
  y = Add(Get2<0>(u), Get2<1>(u));

  y = Mul(Mul(Set(df, -2.0 / 3.0), y), z);
  v = AddDD(df, MulDD(df, z, z), y);
  v = MulDD(df, v, d);
  v = MulDD(df, v, q2);
  z = LoadExp2(df, Add(Get2<0>(v), Get2<1>(v)), Sub(qu, Set(di, 2048)));

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  z = IfThenElse(IsInf(d), MulSignBit(df, Set(df, InfDouble), Get2<0>(q2)), z);
  z = IfThenElse(Eq(d, Set(df, 0)), BitCast(df, SignBit(df, Get2<0>(q2))), z);
#else
  z = IfThenElse(IsInf(s), MulSignBit(df, Set(df, InfDouble), s), z);
  z = IfThenElse(Eq(s, Set(df, 0)), MulSignBit(df, Set(df, 0), s), z);
#endif
  
  return z;
}

// Computes sqrt(x^2 + y^2) with 0.5 ULP accuracy
// Translated from libm/sleefsimddp.c:3222 xhypot_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Hypot(const D df, Vec<D> x, Vec<D> y) {
  x = Abs(x);
  y = Abs(y);
  Vec<D> min = Min(x, y), n = min;
  Vec<D> max = Max(x, y), d = max;

  Mask<D> o = Lt(max, Set(df, DoubleMin));
  n = IfThenElse(o, Mul(n, Set(df, UINT64_C(1) << 54)), n);
  d = IfThenElse(o, Mul(d, Set(df, UINT64_C(1) << 54)), d);

  Vec2<D> t = DivDD(df, Create2(df, n, Set(df, 0)), Create2(df, d, Set(df, 0)));
  t = MulDD(df, SqrtDD(df, AddDD(df, SquareDD(df, t), Set(df, 1))), max);
  Vec<D> ret = Add(Get2<0>(t), Get2<1>(t));
  ret = IfThenElse(IsNaN(ret), Set(df, InfDouble), ret);
  ret = IfThenElse(Eq(min, Set(df, 0)), max, ret);
  ret = IfThenElse(Or(IsNaN(x), IsNaN(y)), Set(df, NanDouble), ret);
  ret = IfThenElse(Or(Eq(x, Set(df, InfDouble)), Eq(y, Set(df, InfDouble))), Set(df, InfDouble), ret);

  return ret;
}

// Computes sqrt(x^2 + y^2) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:3243 xhypot_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) HypotFast(const D df, Vec<D> x, Vec<D> y) {
  x = Abs(x);
  y = Abs(y);
  Vec<D> min = Min(x, y);
  Vec<D> max = Max(x, y);

  Vec<D> t = Div(min, max);
  Vec<D> ret = Mul(max, Sqrt(MulAdd(t, t, Set(df, 1))));
  ret = IfThenElse(Eq(min, Set(df, 0)), max, ret);
  ret = IfThenElse(Or(IsNaN(x), IsNaN(y)), Set(df, NanDouble), ret);
  ret = IfThenElse(Or(Eq(x, Set(df, InfDouble)), Eq(y, Set(df, InfDouble))), Set(df, InfDouble), ret);

  return ret;
}

// Computes x^y with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2358 xpow
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Pow(const D df, Vec<D> x, Vec<D> y) {
  RebindToUnsigned<D> du;
  
Mask<D> yisint = IsInt(df, y);
  Mask<D> yisodd = And(IsOdd(df, y), yisint);

  Vec2<D> d = MulDD(df, LogDD(df, Abs(x)), y);
  Vec<D> result = ExpDD_double(df, d);
  result = IfThenElse(Gt(Get2<0>(d), Set(df, 709.78271114955742909217217426)), Set(df, InfDouble), result);

  result = Mul(result, IfThenElse(Gt(x, Set(df, 0)), Set(df, 1), IfThenElse(yisint, IfThenElse(yisodd, Set(df, -1.0), Set(df, 1)), Set(df, NanDouble))));

  Vec<D> efx = MulSignBit(df, Sub(Abs(x), Set(df, 1)), y);

  result = IfThenElse(IsInf(y), BitCast(df, IfThenZeroElse(RebindMask(du, Lt(efx, Set(df, 0.0))), BitCast(du, IfThenElse(Eq(efx, Set(df, 0.0)), Set(df, 1.0), Set(df, InfDouble))))), result);

  result = IfThenElse(Or(IsInf(x), Eq(x, Set(df, 0.0))), MulSignBit(df, IfThenElse(Xor(SignBitMask(df, y), Eq(x, Set(df, 0.0))), Set(df, 0), Set(df, InfDouble)),
					      IfThenElse(yisodd, x, Set(df, 1))), result);

  result = BitCast(df, IfThenElse(RebindMask(du, Or(IsNaN(x), IsNaN(y))), Set(du, -1), BitCast(du, result)));

  result = IfThenElse(Or(Eq(y, Set(df, 0)), Eq(x, Set(df, 1))), Set(df, 1), result);

  return result;

}

// Computes sin(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:521 xsin_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Sin(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> u;
  Vec2<D> s, t, x;
  Vec<RebindToSigned<D>> ql;

  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2));
  Vec<D> dql = Round(Mul(d, Set(df, OneOverPi)));
  ql = ConvertTo(di, Round(dql));
  u = MulAdd(dql, Set(df, -PiA2), d);
  x = AddFastDD(df, u, Mul(dql, Set(df, -PiB2)));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    Vec<D> dqh = Trunc(Mul(d, Set(df, OneOverPi / (1 << 24))));
    dqh = Mul(dqh, Set(df, 1 << 24));
    const Vec<D> dql = Round(MulSub(d, Set(df, OneOverPi), dqh));

    u = MulAdd(dqh, Set(df, -PiA), d);
    s = AddFastDD(df, u, Mul(dql, Set(df, -PiA)));
    s = AddDD(df, s, Mul(dqh, Set(df, -PiB)));
    s = AddDD(df, s, Mul(dql, Set(df, -PiB)));
    s = AddDD(df, s, Mul(dqh, Set(df, -PiC)));
    s = AddDD(df, s, Mul(dql, Set(df, -PiC)));
    s = AddFastDD(df, s, Mul(Add(dqh, dql), Set(df, -PiD)));

    ql = IfThenElse(RebindMask(di, g), ql, ConvertTo(di, Round(dql)));
    x = IfThenElse(df, g, x, s);
    g = Lt(Abs(d), Set(df, TrigRangeMax));

    if (!HWY_LIKELY(AllTrue(df, g))) {
      Vec3<D> ddi = PayneHanekReduction_d(df, d);
      Vec<RebindToSigned<D>> ql2 = And(BitCast(di, Get3<2>(ddi)), Set(di, 3));
      ql2 = Add(Add(ql2, ql2), IfThenElse(RebindMask(di, Gt(Get2<0>(Create2(df, Get3<0>(ddi), Get3<1>(ddi))), Set(df, 0))), Set(di, 2), Set(di, 1)));
      ql2 = ShiftRight<2>(ql2);
      Mask<D> o = RebindMask(df, Eq(And(BitCast(di, Get3<2>(ddi)), Set(di, 1)), Set(di, 1)));
      Vec2<D> t = Create2(df, MulSignBit(df, Set(df, -3.141592653589793116 * 0.5), Get2<0>(Create2(df, Get3<0>(ddi), Get3<1>(ddi)))), MulSignBit(df, Set(df, -1.2246467991473532072e-16 * 0.5), Get2<0>(Create2(df, Get3<0>(ddi), Get3<1>(ddi)))));
      t = AddDD(df, Create2(df, Get3<0>(ddi), Get3<1>(ddi)), t);
      ddi = Set3<0>(Set3<1>(ddi, Get2<1>(IfThenElse(df, o, t, Create2(df, Get3<0>(ddi), Get3<1>(ddi))))), Get2<0>(IfThenElse(df, o, t, Create2(df, Get3<0>(ddi), Get3<1>(ddi)))));
      s = NormalizeDD(df, Create2(df, Get3<0>(ddi), Get3<1>(ddi)));
      ql = IfThenElse(RebindMask(di, g), ql, ql2);
      x = IfThenElse(df, g, x, s);
      x = Set2<0>(x, BitCast(df, IfThenElse(RebindMask(du, Or(IsInf(d), IsNaN(d))), Set(du, -1), BitCast(du, Get2<0>(x)))));
    }
  }
  
  t = x;
  s = SquareDD(df, x);

  Vec<D> s2 = Mul(Get2<0>(s), Get2<0>(s)), s4 = Mul(s2, s2);
  u = Estrin(Get2<0>(s), s2, s4, Set(df, -0.000198412698412046454654947), Set(df, 2.75573192104428224777379e-06), Set(df, -2.5052106814843123359368e-08), Set(df, 1.60589370117277896211623e-10), Set(df, -7.6429259411395447190023e-13), Set(df, 2.72052416138529567917983e-15));
  u = MulAdd(u, Get2<0>(s), Set(df, 0.00833333333333318056201922));

  x = AddFastDD(df, Set(df, 1), MulDD(df, AddFastDD(df, Set(df, -0.166666666666666657414808), Mul(u, Get2<0>(s))), s));
  u = MulDD_double(df, t, x);

  u = BitCast(df, Xor(IfThenElseZero(RebindMask(du, RebindMask(df, Eq(And(ql, Set(di, 1)), Set(di, 1)))), BitCast(du, Set(df, -0.0))), BitCast(du, u)));

  u = IfThenElse(Eq(d, Set(df, 0)), d, u);
  
  return u; // #if !defined(DETERMINISTIC)
}

// Computes cos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:787 xcos_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Cos(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> u;
  Vec2<D> s, t, x;
  Vec<RebindToSigned<D>> ql;

  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2));
  Vec<D> dql = Round(MulAdd(d, Set(df, OneOverPi), Set(df, -0.5)));
  dql = MulAdd(Set(df, 2), dql, Set(df, 1));
  ql = ConvertTo(di, Round(dql));
  x = AddDD(df, d, Mul(dql, Set(df, -PiA2*0.5)));
  x = AddFastDD(df, x, Mul(dql, Set(df, -PiB2*0.5)));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    Vec<D> dqh = Trunc(MulAdd(d, Set(df, OneOverPi / (1 << 23)), Set(df, -OneOverPi / (1 << 24))));
    Vec<RebindToSigned<D>> ql2 = ConvertTo(di, Round(Add(Mul(d, Set(df, OneOverPi)), MulAdd(dqh, Set(df, -(1 << 23)), Set(df, -0.5)))));
    dqh = Mul(dqh, Set(df, 1 << 24));
    ql2 = Add(Add(ql2, ql2), Set(di, 1));
    const Vec<D> dql = ConvertTo(df, ql2);

    u = MulAdd(dqh, Set(df, -PiA * 0.5), d);
    s = AddDD(df, u, Mul(dql, Set(df, -PiA*0.5)));
    s = AddDD(df, s, Mul(dqh, Set(df, -PiB*0.5)));
    s = AddDD(df, s, Mul(dql, Set(df, -PiB*0.5)));
    s = AddDD(df, s, Mul(dqh, Set(df, -PiC*0.5)));
    s = AddDD(df, s, Mul(dql, Set(df, -PiC*0.5)));
    s = AddFastDD(df, s, Mul(Add(dqh, dql), Set(df, -PiD*0.5)));

    ql = IfThenElse(RebindMask(di, g), ql, ql2);
    x = IfThenElse(df, g, x, s);
    g = Lt(Abs(d), Set(df, TrigRangeMax));

    if (!HWY_LIKELY(AllTrue(df, g))) {
      Vec3<D> ddi = PayneHanekReduction_d(df, d);
      Vec<RebindToSigned<D>> ql2 = And(BitCast(di, Get3<2>(ddi)), Set(di, 3));
      ql2 = Add(Add(ql2, ql2), IfThenElse(RebindMask(di, Gt(Get2<0>(Create2(df, Get3<0>(ddi), Get3<1>(ddi))), Set(df, 0))), Set(di, 8), Set(di, 7)));
      ql2 = ShiftRight<1>(ql2);
      Mask<D> o = RebindMask(df, Eq(And(BitCast(di, Get3<2>(ddi)), Set(di, 1)), Set(di, 0)));
      Vec<D> y = IfThenElse(Gt(Get2<0>(Create2(df, Get3<0>(ddi), Get3<1>(ddi))), Set(df, 0)), Set(df, 0), Set(df, -1));
      Vec2<D> t = Create2(df, MulSignBit(df, Set(df, -3.141592653589793116 * 0.5), y), MulSignBit(df, Set(df, -1.2246467991473532072e-16 * 0.5), y));
      t = AddDD(df, Create2(df, Get3<0>(ddi), Get3<1>(ddi)), t);
      ddi = Set3<0>(Set3<1>(ddi, Get2<1>(IfThenElse(df, o, t, Create2(df, Get3<0>(ddi), Get3<1>(ddi))))), Get2<0>(IfThenElse(df, o, t, Create2(df, Get3<0>(ddi), Get3<1>(ddi)))));
      s = NormalizeDD(df, Create2(df, Get3<0>(ddi), Get3<1>(ddi)));
      ql = IfThenElse(RebindMask(di, g), ql, ql2);
      x = IfThenElse(df, g, x, s);
      x = Set2<0>(x, BitCast(df, IfThenElse(RebindMask(du, Or(IsInf(d), IsNaN(d))), Set(du, -1), BitCast(du, Get2<0>(x)))));
    }
  }
  
  t = x;
  s = SquareDD(df, x);

  Vec<D> s2 = Mul(Get2<0>(s), Get2<0>(s)), s4 = Mul(s2, s2);
  u = Estrin(Get2<0>(s), s2, s4, Set(df, -0.000198412698412046454654947), Set(df, 2.75573192104428224777379e-06), Set(df, -2.5052106814843123359368e-08), Set(df, 1.60589370117277896211623e-10), Set(df, -7.6429259411395447190023e-13), Set(df, 2.72052416138529567917983e-15));
  u = MulAdd(u, Get2<0>(s), Set(df, 0.00833333333333318056201922));

  x = AddFastDD(df, Set(df, 1), MulDD(df, AddFastDD(df, Set(df, -0.166666666666666657414808), Mul(u, Get2<0>(s))), s));
  u = MulDD_double(df, t, x);
  
  u = BitCast(df, Xor(IfThenElseZero(RebindMask(du, RebindMask(df, Eq(And(ql, Set(di, 2)), Set(di, 0)))), BitCast(du, Set(df, -0.0))), BitCast(du, u)));
  
  return u; // #if !defined(DETERMINISTIC)
}

// Computes tan(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:1645 xtan_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Tan(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> u;
  Vec2<D> s, t, x, y;
  Mask<D> o;
  Vec<RebindToSigned<D>> ql;
  
  const Vec<D> dql = Round(Mul(d, Set(df, 2 * OneOverPi)));
  ql = ConvertTo(di, Round(dql));
  u = MulAdd(dql, Set(df, -PiA2*0.5), d);
  s = AddFastDD(df, u, Mul(dql, Set(df, -PiB2*0.5)));
  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    Vec<D> dqh = Trunc(Mul(d, Set(df, 2*OneOverPi / (1 << 24))));
    dqh = Mul(dqh, Set(df, 1 << 24));
    x = AddDD(df, MulDD(df, Create2(df, Set(df, TwoOverPiH), Set(df, TwoOverPiL)), d),
			  Sub(IfThenElse(Lt(d, Set(df, 0)), Set(df, -0.5), Set(df, 0.5)), dqh));
    const Vec<D> dql = Trunc(Add(Get2<0>(x), Get2<1>(x)));

    u = MulAdd(dqh, Set(df, -PiA * 0.5), d);
    x = AddFastDD(df, u, Mul(dql, Set(df, -PiA*0.5)));
    x = AddDD(df, x, Mul(dqh, Set(df, -PiB*0.5)));
    x = AddDD(df, x, Mul(dql, Set(df, -PiB*0.5)));
    x = AddDD(df, x, Mul(dqh, Set(df, -PiC*0.5)));
    x = AddDD(df, x, Mul(dql, Set(df, -PiC*0.5)));
    x = AddFastDD(df, x, Mul(Add(dqh, dql), Set(df, -PiD*0.5)));

    ql = IfThenElse(RebindMask(di, g), ql, ConvertTo(di, Round(dql)));
    s = IfThenElse(df, g, s, x);
    g = Lt(Abs(d), Set(df, TrigRangeMax));

    if (!HWY_LIKELY(AllTrue(df, g))) {
      Vec3<D> ddi = PayneHanekReduction_d(df, d);
      x = Create2(df, Get3<0>(ddi), Get3<1>(ddi));
      o = Or(IsInf(d), IsNaN(d));
      x = Set2<0>(x, BitCast(df, IfThenElse(RebindMask(du, o), Set(du, -1), BitCast(du, Get2<0>(x)))));
      x = Set2<1>(x, BitCast(df, IfThenElse(RebindMask(du, o), Set(du, -1), BitCast(du, Get2<1>(x)))));

      ql = IfThenElse(RebindMask(di, g), ql, BitCast(di, Get3<2>(ddi)));
      s = IfThenElse(df, g, s, x);
    }
  }
  
  t = ScaleDD(df, s, Set(df, 0.5));
  s = SquareDD(df, t);

  Vec<D> s2 = Mul(Get2<0>(s), Get2<0>(s)), s4 = Mul(s2, s2);
  u = Estrin(Get2<0>(s), s2, s4, Set(df, +0.1333333333330500581e+0), Set(df, +0.5396825399517272970e-1), Set(df, +0.2186948728185535498e-1), Set(df, +0.8863268409563113126e-2), Set(df, +0.3591611540792499519e-2), Set(df, +0.1460781502402784494e-2), Set(df, +0.5619219738114323735e-3), Set(df, +0.3245098826639276316e-3));

  u = MulAdd(u, Get2<0>(s), Set(df, +0.3333333333333343695e+0));
  x = AddFastDD(df, t, MulDD(df, MulDD(df, s, t), u));

  y = AddFastDD(df, Set(df, -1), SquareDD(df, x));
  x = ScaleDD(df, x, Set(df, -2));

  o = RebindMask(df, Eq(And(ql, Set(di, 1)), Set(di, 1)));

  x = DivDD(df, IfThenElse(df, o, NegDD(df, y), x),
			IfThenElse(df, o, x, y));

  u = Add(Get2<0>(x), Get2<1>(x));

  u = IfThenElse(Eq(d, Set(df, 0)), d, u);

  return u; // #if !defined(DETERMINISTIC)
}

// Computes sin(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:382 xsin
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) SinFast(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
// This is the deterministic implementation of sin function. Returned
// values from deterministic functions are bitwise consistent across
// all platforms. The function name xsin will be renamed to
// Sleef_cinz_sind2_u35sse2 with renamesse2.h, for example. The
// renaming by rename*.h is switched according to DETERMINISTIC macro.
  Vec<D> u, s, r = d;
  Vec<RebindToSigned<D>> ql;

  Vec<D> dql = Round(Mul(d, Set(df, OneOverPi)));
  ql = ConvertTo(di, Round(dql));
  d = MulAdd(dql, Set(df, -PiA2), d);
  d = MulAdd(dql, Set(df, -PiB2), d);
  Mask<D> g = Lt(Abs(r), Set(df, TrigRangeMax2));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    Vec<D> dqh = Trunc(Mul(r, Set(df, OneOverPi / (1 << 24))));
    dqh = Mul(dqh, Set(df, 1 << 24));
    Vec<D> dql = Round(MulSub(r, Set(df, OneOverPi), dqh));

    u = MulAdd(dqh, Set(df, -PiA), r);
    u = MulAdd(dql, Set(df, -PiA), u);
    u = MulAdd(dqh, Set(df, -PiB), u);
    u = MulAdd(dql, Set(df, -PiB), u);
    u = MulAdd(dqh, Set(df, -PiC), u);
    u = MulAdd(dql, Set(df, -PiC), u);
    u = MulAdd(Add(dqh, dql), Set(df, -PiD), u);

    ql = IfThenElse(RebindMask(di, g), ql, ConvertTo(di, Round(dql)));
    d = IfThenElse(g, d, u);
    g = Lt(Abs(r), Set(df, TrigRangeMax));

    if (!HWY_LIKELY(AllTrue(df, g))) {
      Vec3<D> ddi = PayneHanekReduction_d(df, r);
      Vec<RebindToSigned<D>> ql2 = And(BitCast(di, Get3<2>(ddi)), Set(di, 3));
      ql2 = Add(Add(ql2, ql2), IfThenElse(RebindMask(di, Gt(Get2<0>(Create2(df, Get3<0>(ddi), Get3<1>(ddi))), Set(df, 0))), Set(di, 2), Set(di, 1)));
      ql2 = ShiftRight<2>(ql2);
      Mask<D> o = RebindMask(df, Eq(And(BitCast(di, Get3<2>(ddi)), Set(di, 1)), Set(di, 1)));
      Vec2<D> x = Create2(df, MulSignBit(df, Set(df, -3.141592653589793116 * 0.5), Get2<0>(Create2(df, Get3<0>(ddi), Get3<1>(ddi)))), MulSignBit(df, Set(df, -1.2246467991473532072e-16 * 0.5), Get2<0>(Create2(df, Get3<0>(ddi), Get3<1>(ddi)))));
      x = AddDD(df, Create2(df, Get3<0>(ddi), Get3<1>(ddi)), x);
      ddi = Set3<0>(Set3<1>(ddi, Get2<1>(IfThenElse(df, o, x, Create2(df, Get3<0>(ddi), Get3<1>(ddi))))), Get2<0>(IfThenElse(df, o, x, Create2(df, Get3<0>(ddi), Get3<1>(ddi)))));
      u = Add(Get2<0>(Create2(df, Get3<0>(ddi), Get3<1>(ddi))), Get2<1>(Create2(df, Get3<0>(ddi), Get3<1>(ddi))));
      ql = IfThenElse(RebindMask(di, g), ql, ql2);
      d = IfThenElse(g, d, u);
      d = BitCast(df, IfThenElse(RebindMask(du, Or(IsInf(r), IsNaN(r))), Set(du, -1), BitCast(du, d)));
    }
  }

  s = Mul(d, d);

  d = BitCast(df, Xor(IfThenElseZero(RebindMask(du, RebindMask(df, Eq(And(ql, Set(di, 1)), Set(di, 1)))), BitCast(du, Set(df, -0.0))), BitCast(du, d)));

  Vec<D> s2 = Mul(s, s), s4 = Mul(s2, s2);
  u = Estrin(s, s2, s4, Set(df, 0.00833333333333332974823815), Set(df, -0.000198412698412696162806809), Set(df, 2.75573192239198747630416e-06), Set(df, -2.50521083763502045810755e-08), Set(df, 1.60590430605664501629054e-10), Set(df, -7.64712219118158833288484e-13), Set(df, 2.81009972710863200091251e-15), Set(df, -7.97255955009037868891952e-18));
  u = MulAdd(u, s, Set(df, -0.166666666666666657414808));

  u = Add(Mul(s, Mul(u, d)), d);

  u = IfThenElse(Eq(r, Set(df, -0.0)), r, u);
  
  return u; // #if !defined(DETERMINISTIC)
}

// Computes cos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:652 xcos
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) CosFast(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> u, s, r = d;
  Vec<RebindToSigned<D>> ql;

  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2));
  Vec<D> dql = MulAdd(Set(df, 2), Round(MulAdd(d, Set(df, OneOverPi), Set(df, -0.5))), Set(df, 1));
  ql = ConvertTo(di, Round(dql));
  d = MulAdd(dql, Set(df, -PiA2 * 0.5), d);
  d = MulAdd(dql, Set(df, -PiB2 * 0.5), d);

  if (!HWY_LIKELY(AllTrue(df, g))) {
    Vec<D> dqh = Trunc(MulAdd(r, Set(df, OneOverPi / (1 << 23)), Set(df, -OneOverPi / (1 << 24))));
    Vec<RebindToSigned<D>> ql2 = ConvertTo(di, Round(Add(Mul(r, Set(df, OneOverPi)), MulAdd(dqh, Set(df, -(1 << 23)), Set(df, -0.5)))));
    dqh = Mul(dqh, Set(df, 1 << 24));
    ql2 = Add(Add(ql2, ql2), Set(di, 1));
    Vec<D> dql = ConvertTo(df, ql2);

    u = MulAdd(dqh, Set(df, -PiA * 0.5), r);
    u = MulAdd(dql, Set(df, -PiA * 0.5), u);
    u = MulAdd(dqh, Set(df, -PiB * 0.5), u);
    u = MulAdd(dql, Set(df, -PiB * 0.5), u);
    u = MulAdd(dqh, Set(df, -PiC * 0.5), u);
    u = MulAdd(dql, Set(df, -PiC * 0.5), u);
    u = MulAdd(Add(dqh, dql), Set(df, -PiD * 0.5), u);

    ql = IfThenElse(RebindMask(di, g), ql, ql2);
    d = IfThenElse(g, d, u);
    g = Lt(Abs(r), Set(df, TrigRangeMax));

    if (!HWY_LIKELY(AllTrue(df, g))) {
      Vec3<D> ddi = PayneHanekReduction_d(df, r);
      Vec<RebindToSigned<D>> ql2 = And(BitCast(di, Get3<2>(ddi)), Set(di, 3));
      ql2 = Add(Add(ql2, ql2), IfThenElse(RebindMask(di, Gt(Get2<0>(Create2(df, Get3<0>(ddi), Get3<1>(ddi))), Set(df, 0))), Set(di, 8), Set(di, 7)));
      ql2 = ShiftRight<1>(ql2);
      Mask<D> o = RebindMask(df, Eq(And(BitCast(di, Get3<2>(ddi)), Set(di, 1)), Set(di, 0)));
      Vec<D> y = IfThenElse(Gt(Get2<0>(Create2(df, Get3<0>(ddi), Get3<1>(ddi))), Set(df, 0)), Set(df, 0), Set(df, -1));
      Vec2<D> x = Create2(df, MulSignBit(df, Set(df, -3.141592653589793116 * 0.5), y), MulSignBit(df, Set(df, -1.2246467991473532072e-16 * 0.5), y));
      x = AddDD(df, Create2(df, Get3<0>(ddi), Get3<1>(ddi)), x);
      ddi = Set3<0>(Set3<1>(ddi, Get2<1>(IfThenElse(df, o, x, Create2(df, Get3<0>(ddi), Get3<1>(ddi))))), Get2<0>(IfThenElse(df, o, x, Create2(df, Get3<0>(ddi), Get3<1>(ddi)))));
      u = Add(Get2<0>(Create2(df, Get3<0>(ddi), Get3<1>(ddi))), Get2<1>(Create2(df, Get3<0>(ddi), Get3<1>(ddi))));
      ql = IfThenElse(RebindMask(di, g), ql, ql2);
      d = IfThenElse(g, d, u);
      d = BitCast(df, IfThenElse(RebindMask(du, Or(IsInf(r), IsNaN(r))), Set(du, -1), BitCast(du, d)));
    }
  }

  s = Mul(d, d);

  d = BitCast(df, Xor(IfThenElseZero(RebindMask(du, RebindMask(df, Eq(And(ql, Set(di, 2)), Set(di, 0)))), BitCast(du, Set(df, -0.0))), BitCast(du, d)));

  Vec<D> s2 = Mul(s, s), s4 = Mul(s2, s2);
  u = Estrin(s, s2, s4, Set(df, 0.00833333333333332974823815), Set(df, -0.000198412698412696162806809), Set(df, 2.75573192239198747630416e-06), Set(df, -2.50521083763502045810755e-08), Set(df, 1.60590430605664501629054e-10), Set(df, -7.64712219118158833288484e-13), Set(df, 2.81009972710863200091251e-15), Set(df, -7.97255955009037868891952e-18));
  u = MulAdd(u, s, Set(df, -0.166666666666666657414808));

  u = Add(Mul(s, Mul(u, d)), d);
  
  return u; // #if !defined(DETERMINISTIC)
}

// Computes tan(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:1517 xtan
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) TanFast(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> u, s, x, y;
  Mask<D> o;
  Vec<RebindToSigned<D>> ql;

  Vec<D> dql = Round(Mul(d, Set(df, 2 * OneOverPi)));
  ql = ConvertTo(di, Round(dql));
  s = MulAdd(dql, Set(df, -PiA2 * 0.5), d);
  s = MulAdd(dql, Set(df, -PiB2 * 0.5), s);
  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    Vec<D> dqh = Trunc(Mul(d, Set(df, 2*OneOverPi / (1 << 24))));
    dqh = Mul(dqh, Set(df, 1 << 24));
    Vec<D> dql = Round(Sub(Mul(d, Set(df, 2*OneOverPi)), dqh));

    u = MulAdd(dqh, Set(df, -PiA * 0.5), d);
    u = MulAdd(dql, Set(df, -PiA * 0.5), u);
    u = MulAdd(dqh, Set(df, -PiB * 0.5), u);
    u = MulAdd(dql, Set(df, -PiB * 0.5), u);
    u = MulAdd(dqh, Set(df, -PiC * 0.5), u);
    u = MulAdd(dql, Set(df, -PiC * 0.5), u);
    u = MulAdd(Add(dqh, dql), Set(df, -PiD * 0.5), u);

    ql = IfThenElse(RebindMask(di, g), ql, ConvertTo(di, Round(dql)));
    s = IfThenElse(g, s, u);
    g = Lt(Abs(d), Set(df, 1e+6));

    if (!HWY_LIKELY(AllTrue(df, g))) {
      Vec3<D> ddi = PayneHanekReduction_d(df, d);
      Vec<RebindToSigned<D>> ql2 = BitCast(di, Get3<2>(ddi));
      u = Add(Get2<0>(Create2(df, Get3<0>(ddi), Get3<1>(ddi))), Get2<1>(Create2(df, Get3<0>(ddi), Get3<1>(ddi))));
      u = BitCast(df, IfThenElse(RebindMask(du, Or(IsInf(d), IsNaN(d))), Set(du, -1), BitCast(du, u)));

      ql = IfThenElse(RebindMask(di, g), ql, ql2);
      s = IfThenElse(g, s, u);
    }
  }

  x = Mul(s, Set(df, 0.5));
  s = Mul(x, x);

  Vec<D> s2 = Mul(s, s), s4 = Mul(s2, s2);
  u = Estrin(s, s2, s4, Set(df, +0.1333333333330500581e+0), Set(df, +0.5396825399517272970e-1), Set(df, +0.2186948728185535498e-1), Set(df, +0.8863268409563113126e-2), Set(df, +0.3591611540792499519e-2), Set(df, +0.1460781502402784494e-2), Set(df, +0.5619219738114323735e-3), Set(df, +0.3245098826639276316e-3));

  u = MulAdd(u, s, Set(df, +0.3333333333333343695e+0));
  u = MulAdd(s, Mul(u, x), x);

  y = MulAdd(u, u, Set(df, -1));
  x = Mul(u, Set(df, -2));

  o = RebindMask(df, Eq(And(ql, Set(di, 1)), Set(di, 1)));
  u = Div(IfThenElse(o, Neg(y), x), IfThenElse(o, x, y));
  u = IfThenElse(Eq(d, Set(df, 0)), d, u);
  
  return u; // #if !defined(DETERMINISTIC)
}

// Computes sinh(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2435 xsinh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Sinh(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> y = Abs(x);
  Vec2<D> d = ExpDD(df, Create2(df, y, Set(df, 0)));
  d = SubDD(df, d, RecDD(df, d));
  y = Mul(Add(Get2<0>(d), Get2<1>(d)), Set(df, 0.5));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, 710)), IsNaN(y)), Set(df, InfDouble), y);
  y = MulSignBit(df, y, x);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

// Computes cosh(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2448 xcosh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Cosh(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> y = Abs(x);
  Vec2<D> d = ExpDD(df, Create2(df, y, Set(df, 0)));
  d = AddFastDD(df, d, RecDD(df, d));
  y = Mul(Add(Get2<0>(d), Get2<1>(d)), Set(df, 0.5));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, 710)), IsNaN(y)), Set(df, InfDouble), y);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

// Computes tanh(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2460 xtanh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Tanh(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> y = Abs(x);
  Vec2<D> d = ExpDD(df, Create2(df, y, Set(df, 0)));
  Vec2<D> e = RecDD(df, d);
  d = DivDD(df, AddDD(df, d, NegDD(df, e)), AddDD(df, d, e));
  y = Add(Get2<0>(d), Get2<1>(d));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, 18.714973875)), IsNaN(y)), Set(df, 1.0), y);
  y = MulSignBit(df, y, x);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

// Computes sinh(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:2474 xsinh_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) SinhFast(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> e = Expm1k(df, Abs(x));

  Vec<D> y = Div(Add(e, Set(df, 2)), Add(e, Set(df, 1)));
  y = Mul(y, Mul(Set(df, 0.5), e));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, 709)), IsNaN(y)), Set(df, InfDouble), y);
  y = MulSignBit(df, y, x);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

// Computes cosh(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:2487 xcosh_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) CoshFast(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> e = sleef::Exp(df, Abs(x));
  Vec<D> y = MulAdd(Set(df, 0.5), e, Div(Set(df, 0.5), e));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, 709)), IsNaN(y)), Set(df, InfDouble), y);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

// Computes tanh(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:2497 xtanh_u35
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) TanhFast(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> d = Expm1k(df, Mul(Set(df, 2), Abs(x)));
  Vec<D> y = Div(d, Add(Set(df, 2), d));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, 18.714973875)), IsNaN(y)), Set(df, 1.0), y);
  y = MulSignBit(df, y, x);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

// Computes acos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2007 xacos_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Acos(const D df, Vec<D> d) {
  Mask<D> o = Lt(Abs(d), Set(df, 0.5));
  Vec<D> x2 = IfThenElse(o, Mul(d, d), Mul(Sub(Set(df, 1), Abs(d)), Set(df, 0.5))), u;
  Vec2<D> x = IfThenElse(df, o, Create2(df, Abs(d), Set(df, 0)), SqrtDD(df, x2));
  x = IfThenElse(df, Eq(Abs(d), Set(df, 1.0)), Create2(df, Set(df, 0), Set(df, 0)), x);

  Vec<D> x4 = Mul(x2, x2), x8 = Mul(x4, x4), x16 = Mul(x8, x8);
  u = Estrin(x2, x4, x8, x16, Set(df, +0.1666666666666497543e+0), Set(df, +0.7500000000378581611e-1), Set(df, +0.4464285681377102438e-1), Set(df, +0.3038195928038132237e-1), Set(df, +0.2237176181932048341e-1), Set(df, +0.1735956991223614604e-1), Set(df, +0.1388715184501609218e-1), Set(df, +0.1215360525577377331e-1), Set(df, +0.6606077476277170610e-2), Set(df, +0.1929045477267910674e-1), Set(df, -0.1581918243329996643e-1), Set(df, +0.3161587650653934628e-1));

  u = Mul(u, Mul(x2, Get2<0>(x)));

  Vec2<D> y = SubDD(df, Create2(df, Set(df, 3.141592653589793116/2), Set(df, 1.2246467991473532072e-16/2)),
				 AddFastDD(df, MulSignBit(df, Get2<0>(x), d), MulSignBit(df, u, d)));
  x = AddFastDD(df, x, u);
  
  y = IfThenElse(df, o, y, ScaleDD(df, x, Set(df, 2)));
  
  y = IfThenElse(df, AndNot(o, Lt(d, Set(df, 0))),
			  SubDD(df, Create2(df, Set(df, 3.141592653589793116), Set(df, 1.2246467991473532072e-16)), y), y);

  return Add(Get2<0>(y), Get2<1>(y));
}

// Computes asin(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:1945 xasin_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Asin(const D df, Vec<D> d) {
  Mask<D> o = Lt(Abs(d), Set(df, 0.5));
  Vec<D> x2 = IfThenElse(o, Mul(d, d), Mul(Sub(Set(df, 1), Abs(d)), Set(df, 0.5))), u;
  Vec2<D> x = IfThenElse(df, o, Create2(df, Abs(d), Set(df, 0)), SqrtDD(df, x2));
  x = IfThenElse(df, Eq(Abs(d), Set(df, 1.0)), Create2(df, Set(df, 0), Set(df, 0)), x);

  Vec<D> x4 = Mul(x2, x2), x8 = Mul(x4, x4), x16 = Mul(x8, x8);
  u = Estrin(x2, x4, x8, x16, Set(df, +0.1666666666666497543e+0), Set(df, +0.7500000000378581611e-1), Set(df, +0.4464285681377102438e-1), Set(df, +0.3038195928038132237e-1), Set(df, +0.2237176181932048341e-1), Set(df, +0.1735956991223614604e-1), Set(df, +0.1388715184501609218e-1), Set(df, +0.1215360525577377331e-1), Set(df, +0.6606077476277170610e-2), Set(df, +0.1929045477267910674e-1), Set(df, -0.1581918243329996643e-1), Set(df, +0.3161587650653934628e-1));

  u = Mul(u, Mul(x2, Get2<0>(x)));

  Vec2<D> y = SubDD(df, SubDD(df, Create2(df, Set(df, 3.141592653589793116/4), Set(df, 1.2246467991473532072e-16/4)), x), u);
  
  Vec<D> r = IfThenElse(o, Add(u, Get2<0>(x)), Mul(Add(Get2<0>(y), Get2<1>(y)), Set(df, 2)));
  return MulSignBit(df, r, d);
}

// Computes asinh(x) with 1 ULP accuracy
// Translated from libm/sleefsimddp.c:2539 xasinh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Asinh(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> y = Abs(x);
  Mask<D> o = Gt(y, Set(df, 1));
  Vec2<D> d;
  
  d = IfThenElse(df, o, RecDD(df, x), Create2(df, y, Set(df, 0)));
  d = SqrtDD(df, AddDD(df, SquareDD(df, d), Set(df, 1)));
  d = IfThenElse(df, o, MulDD(df, d, y), d);

  d = LogFastDD(df, NormalizeDD(df, AddDD(df, d, x)));
  y = Add(Get2<0>(d), Get2<1>(d));
  
  y = IfThenElse(Or(Gt(Abs(x), Set(df, SqrtDoubleMax)), IsNaN(y)), MulSignBit(df, Set(df, InfDouble), x), y);

  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));
  y = IfThenElse(Eq(x, Set(df, -0.0)), Set(df, -0.0), y);
  
  return y;
}

// Computes acos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:1975 xacos
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) AcosFast(const D df, Vec<D> d) {
  Mask<D> o = Lt(Abs(d), Set(df, 0.5));
  Vec<D> x2 = IfThenElse(o, Mul(d, d), Mul(Sub(Set(df, 1), Abs(d)), Set(df, 0.5))), u;
  Vec<D> x = IfThenElse(o, Abs(d), Sqrt(x2));
  x = IfThenElse(Eq(Abs(d), Set(df, 1.0)), Set(df, 0), x);

  Vec<D> x4 = Mul(x2, x2), x8 = Mul(x4, x4), x16 = Mul(x8, x8);
  u = Estrin(x2, x4, x8, x16, Set(df, +0.1666666666666497543e+0), Set(df, +0.7500000000378581611e-1), Set(df, +0.4464285681377102438e-1), Set(df, +0.3038195928038132237e-1), Set(df, +0.2237176181932048341e-1), Set(df, +0.1735956991223614604e-1), Set(df, +0.1388715184501609218e-1), Set(df, +0.1215360525577377331e-1), Set(df, +0.6606077476277170610e-2), Set(df, +0.1929045477267910674e-1), Set(df, -0.1581918243329996643e-1), Set(df, +0.3161587650653934628e-1));

  u = Mul(u, Mul(x2, x));

  Vec<D> y = Sub(Set(df, Pi/2), Add(MulSignBit(df, x, d), MulSignBit(df, u, d)));
  x = Add(x, u);
  Vec<D> r = IfThenElse(o, y, Mul(x, Set(df, 2)));
  return IfThenElse(AndNot(o, Lt(d, Set(df, 0))), Get2<0>(AddFastDD(df, Create2(df, Set(df, 3.141592653589793116), Set(df, 1.2246467991473532072e-16)),
							  Neg(r))), r);
}

// Computes asin(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:1919 xasin
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) AsinFast(const D df, Vec<D> d) {
  Mask<D> o = Lt(Abs(d), Set(df, 0.5));
  Vec<D> x2 = IfThenElse(o, Mul(d, d), Mul(Sub(Set(df, 1), Abs(d)), Set(df, 0.5)));
  Vec<D> x = IfThenElse(o, Abs(d), Sqrt(x2)), u;

  Vec<D> x4 = Mul(x2, x2), x8 = Mul(x4, x4), x16 = Mul(x8, x8);
  u = Estrin(x2, x4, x8, x16, Set(df, +0.1666666666666497543e+0), Set(df, +0.7500000000378581611e-1), Set(df, +0.4464285681377102438e-1), Set(df, +0.3038195928038132237e-1), Set(df, +0.2237176181932048341e-1), Set(df, +0.1735956991223614604e-1), Set(df, +0.1388715184501609218e-1), Set(df, +0.1215360525577377331e-1), Set(df, +0.6606077476277170610e-2), Set(df, +0.1929045477267910674e-1), Set(df, -0.1581918243329996643e-1), Set(df, +0.3161587650653934628e-1));

  u = MulAdd(u, Mul(x, x2), x);
  
  Vec<D> r = IfThenElse(o, u, MulAdd(u, Set(df, -2), Set(df, Pi/2)));
  return MulSignBit(df, r, d);
}

// Computes atan(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:2049 xatan
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) AtanFast(const D df, Vec<D> s) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<D> t, u;
  Vec<RebindToSigned<D>> q;
#if defined(__INTEL_COMPILER) && defined(ENABLE_PURECFMA_SCALAR)
  Vec<D> w = s;
#endif

  q = SignBitOrZero(df, s, Set(di, 2));
  s = Abs(s);

  q = LessThanSelect(df, Set(df, 1), s, Add(q, Set(di, 1)), q);
  s = IfThenElse(Lt(Set(df, 1), s), Div(Set(df, 1.0), s), s);

  t = Mul(s, s);

  Vec<D> t2 = Mul(t, t), t4 = Mul(t2, t2), t8 = Mul(t4, t4), t16 = Mul(t8, t8);
  u = Estrin(t, t2, t4, t8, t16, Set(df, -0.333333333333311110369124), Set(df, 0.199999999996591265594148), Set(df, -0.14285714266771329383765), Set(df, 0.111111105648261418443745), Set(df, -0.090908995008245008229153), Set(df, 0.0769219538311769618355029), Set(df, -0.0666573579361080525984562), Set(df, 0.0587666392926673580854313), Set(df, -0.0523674852303482457616113), Set(df, 0.0466667150077840625632675), Set(df, -0.0407629191276836500001934), Set(df, 0.0337852580001353069993897), Set(df, -0.0254517624932312641616861), Set(df, 0.016599329773529201970117), Set(df, -0.00889896195887655491740809), Set(df, 0.00370026744188713119232403), Set(df, -0.00110611831486672482563471), Set(df, 0.000209850076645816976906797), Set(df, -1.88796008463073496563746e-05));
  
  t = MulAdd(s, Mul(t, u), s);

  t = IfThenElse(RebindMask(df, Eq(And(q, Set(di, 1)), Set(di, 1))), Sub(Set(df, Pi/2), t), t);
  t = BitCast(df, Xor(IfThenElseZero(RebindMask(du, RebindMask(df, Eq(And(q, Set(di, 2)), Set(di, 2)))), BitCast(du, Set(df, -0.0))), BitCast(du, t)));

#if defined(__INTEL_COMPILER) && defined(ENABLE_PURECFMA_SCALAR)
  t = IfThenElse(Eq(w, Set(df, 0)), w, t);
#endif

  return t;
}

// Computes acosh(x) with 1 ULP accuracy
// Translated from libm/sleefsimddp.c:2561 xacosh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Acosh(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec2<D> d = LogFastDD(df, AddDD(df, MulDD(df, SqrtDD(df, AddDD(df, x, Set(df, 1))), SqrtDD(df, AddDD(df, x, Set(df, -1)))), x));
  Vec<D> y = Add(Get2<0>(d), Get2<1>(d));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, SqrtDoubleMax)), IsNaN(y)), Set(df, InfDouble), y);
  y = BitCast(df, IfThenZeroElse(RebindMask(du, Eq(x, Set(df, 1.0))), BitCast(du, y)));

  y = BitCast(df, IfThenElse(RebindMask(du, Lt(x, Set(df, 1.0))), Set(du, -1), BitCast(du, y)));
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));
  
  return y;
}

// Computes atan(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2042 xatan_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Atan(const D df, Vec<D> d) {
  Vec2<D> d2 = ATan2DD(df, Create2(df, Abs(d), Set(df, 0)), Create2(df, Set(df, 1), Set(df, 0)));
  Vec<D> r = Add(Get2<0>(d2), Get2<1>(d2));
  r = IfThenElse(IsInf(d), Set(df, 1.570796326794896557998982), r);
  return MulSignBit(df, r, d);
}

// Computes atanh(x) with 1 ULP accuracy
// Translated from libm/sleefsimddp.c:2576 xatanh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Atanh(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> y = Abs(x);
  Vec2<D> d = LogFastDD(df, DivDD(df, AddDD(df, Set(df, 1), y), AddDD(df, Set(df, 1), Neg(y))));
  y = BitCast(df, IfThenElse(RebindMask(du, Gt(y, Set(df, 1.0))), Set(du, -1), BitCast(du, IfThenElse(Eq(y, Set(df, 1.0)), Set(df, InfDouble), Mul(Add(Get2<0>(d), Get2<1>(d)), Set(df, 0.5))))));

  y = MulSignBit(df, y, x);
  y = BitCast(df, IfThenElse(RebindMask(du, Or(IsInf(x), IsNaN(y))), Set(du, -1), BitCast(du, y)));
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
}

}  // namespace sleef
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_SLEEF_SLEEF_INL_

#if HWY_ONCE
__attribute__((aligned(64)))
const double PayneHanekReductionTable_double[] = {
    // clang-format off
  0.15915494309189531785, 1.7916237278037667488e-17, 2.5454160968749269937e-33, 2.1132476107887107169e-49,
  0.03415494309189533173, 4.0384494702232122736e-18, 1.0046721413651383112e-33, 2.1132476107887107169e-49,
  0.03415494309189533173, 4.0384494702232122736e-18, 1.0046721413651383112e-33, 2.1132476107887107169e-49,
  0.0029049430918953351999, 5.6900251826959904774e-19, 4.1707169171520598517e-35, -2.496415728504571394e-51,
  0.0029049430918953351999, 5.6900251826959904774e-19, 4.1707169171520598517e-35, -2.496415728504571394e-51,
  0.0029049430918953351999, 5.6900251826959904774e-19, 4.1707169171520598517e-35, -2.496415728504571394e-51,
  0.0029049430918953351999, 5.6900251826959904774e-19, 4.1707169171520598517e-35, -2.496415728504571394e-51,
  0.00095181809189533563356, 1.3532164927539732229e-19, -6.4410794381603004826e-36, 1.7634898158762436344e-52,
  0.00095181809189533563356, 1.3532164927539732229e-19, -6.4410794381603004826e-36, 1.7634898158762436344e-52,
  0.00046353684189533574198, 2.6901432026846872871e-20, -4.2254836195018827479e-37, 9.301187206862134399e-54,
  0.00021939621689533574198, 2.6901432026846872871e-20, -4.2254836195018827479e-37, 9.301187206862134399e-54,
  9.7325904395335769087e-05, -2.0362228529073840241e-22, 6.2960434583523738135e-40, 2.6283399642369025999e-57,
  3.6290748145335769087e-05, -2.0362228529073840241e-22, 6.2960434583523738135e-40, 2.6283399642369025999e-57,
  5.7731700203357690874e-06, -2.0362228529073840241e-22, 6.2960434583523738135e-40, 2.6283399642369025999e-57,
  5.7731700203357690874e-06, -2.0362228529073840241e-22, 6.2960434583523738135e-40, 2.6283399642369025999e-57,
  5.7731700203357690874e-06, -2.0362228529073840241e-22, 6.2960434583523738135e-40, 2.6283399642369025999e-57,
  1.9584727547107690874e-06, -2.0362228529073840241e-22, 6.2960434583523738135e-40, 2.6283399642369025999e-57,
  5.1124121898268875627e-08, 8.135951522836682362e-24, 6.2960434583523738135e-40, 2.6283399642369025999e-57,
  5.1124121898268875627e-08, 8.135951522836682362e-24, 6.2960434583523738135e-40, 2.6283399642369025999e-57,
  5.1124121898268875627e-08, 8.135951522836682362e-24, 6.2960434583523738135e-40, 2.6283399642369025999e-57,
  5.1124121898268875627e-08, 8.135951522836682362e-24, 6.2960434583523738135e-40, 2.6283399642369025999e-57,
  5.1124121898268875627e-08, 8.135951522836682362e-24, 6.2960434583523738135e-40, 2.6283399642369025999e-57,
  5.1124121898268875627e-08, 8.135951522836682362e-24, 6.2960434583523738135e-40, 2.6283399642369025999e-57,
  2.1321799510573569745e-08, 1.5185066224124613304e-24, 2.6226236120327253511e-40, 2.6283399642369025999e-57,
  6.4206383167259151492e-09, -1.3585460269359374382e-25, -1.3244127270701094468e-41, -2.4695541513869446866e-57,
  6.4206383167259151492e-09, -1.3585460269359374382e-25, -1.3244127270701094468e-41, -2.4695541513869446866e-57,
  2.6953480182640010867e-09, -1.3585460269359374382e-25, -1.3244127270701094468e-41, -2.4695541513869446866e-57,
  8.3270286903304384868e-10, 7.0940550444663151936e-26, 9.7147467687967058732e-42, 7.9392906424978921242e-59,
  8.3270286903304384868e-10, 7.0940550444663151936e-26, 9.7147467687967058732e-42, 7.9392906424978921242e-59,
  3.6704158172530459087e-10, 7.0940550444663151936e-26, 9.7147467687967058732e-42, 7.9392906424978921242e-59,
  1.3421093807143501366e-10, 1.9241762160098927996e-26, 3.9750282589222551507e-42, 7.9392906424978921242e-59,
  1.7795616244500218596e-11, -1.452834466126541428e-28, -1.5869767474823787636e-44, -2.6168913164368963837e-61,
  1.7795616244500218596e-11, -1.452834466126541428e-28, -1.5869767474823787636e-44, -2.6168913164368963837e-61,
  1.7795616244500218596e-11, -1.452834466126541428e-28, -1.5869767474823787636e-44, -2.6168913164368963837e-61,
  3.2437010161333667893e-12, -1.452834466126541428e-28, -1.5869767474823787636e-44, -2.6168913164368963837e-61,
  3.2437010161333667893e-12, -1.452834466126541428e-28, -1.5869767474823787636e-44, -2.6168913164368963837e-61,
  3.2437010161333667893e-12, -1.452834466126541428e-28, -1.5869767474823787636e-44, -2.6168913164368963837e-61,
  1.4247116125875099096e-12, 2.5861333686050385673e-28, 2.8971783383570358633e-44, -2.6168913164368963837e-61,
  5.1521691081458187359e-13, 5.6664945123924856962e-29, 6.5510079543732854985e-45, -2.6168913164368963837e-61,
  6.0469559928117805118e-14, 6.1778471897801070206e-30, 9.4581409707401690366e-46, 4.9461632249367446986e-62,
  6.0469559928117805118e-14, 6.1778471897801070206e-30, 9.4581409707401690366e-46, 4.9461632249367446986e-62,
  6.0469559928117805118e-14, 6.1778471897801070206e-30, 9.4581409707401690366e-46, 4.9461632249367446986e-62,
  3.6261410673097965595e-15, -1.3304005198798645927e-31, -1.7578597149294783985e-47, 8.4432539107728104262e-64,
  3.6261410673097965595e-15, -1.3304005198798645927e-31, -1.7578597149294783985e-47, 8.4432539107728104262e-64,
  3.6261410673097965595e-15, -1.3304005198798645927e-31, -1.7578597149294783985e-47, 8.4432539107728104262e-64,
  3.6261410673097965595e-15, -1.3304005198798645927e-31, -1.7578597149294783985e-47, 8.4432539107728104262e-64,
  7.3427388509295482183e-17, 1.4871367740953237822e-32, -1.1571307704883330232e-48, -6.7249112515659578102e-65,
  7.3427388509295482183e-17, 1.4871367740953237822e-32, -1.1571307704883330232e-48, -6.7249112515659578102e-65,
  7.3427388509295482183e-17, 1.4871367740953237822e-32, -1.1571307704883330232e-48, -6.7249112515659578102e-65,
  7.3427388509295482183e-17, 1.4871367740953237822e-32, -1.1571307704883330232e-48, -6.7249112515659578102e-65,
  7.3427388509295482183e-17, 1.4871367740953237822e-32, -1.1571307704883330232e-48, -6.7249112515659578102e-65,
  7.3427388509295482183e-17, 1.4871367740953237822e-32, -1.1571307704883330232e-48, -6.7249112515659578102e-65,
  1.7916237278037667488e-17, 2.5454160968749269937e-33, 2.1132476107887107169e-49, 8.7154294504188129325e-66,
  1.7916237278037667488e-17, 2.5454160968749269937e-33, 2.1132476107887107169e-49, 8.7154294504188129325e-66,
  4.0384494702232122736e-18, 1.0046721413651383112e-33, 2.1132476107887107169e-49, 8.7154294504188129325e-66,
  4.0384494702232122736e-18, 1.0046721413651383112e-33, 2.1132476107887107169e-49, 8.7154294504188129325e-66,
  5.6900251826959904774e-19, 4.1707169171520598517e-35, -2.4964157285045710972e-51, -1.866653112309982615e-67,
  5.6900251826959904774e-19, 4.1707169171520598517e-35, -2.4964157285045710972e-51, -1.866653112309982615e-67,
  5.6900251826959904774e-19, 4.1707169171520598517e-35, -2.4964157285045710972e-51, -1.866653112309982615e-67,
  1.3532164927539732229e-19, -6.4410794381603004826e-36, 1.7634898158762432635e-52, 3.5887057810247033998e-68,
  1.3532164927539732229e-19, -6.4410794381603004826e-36, 1.7634898158762432635e-52, 3.5887057810247033998e-68,
  2.6901432026846872871e-20, -4.2254836195018827479e-37, 9.3011872068621332399e-54, 1.113250147552460308e-69,
  2.6901432026846872871e-20, -4.2254836195018827479e-37, 9.3011872068621332399e-54, 1.113250147552460308e-69,
  2.6901432026846872871e-20, -4.2254836195018827479e-37, 9.3011872068621332399e-54, 1.113250147552460308e-69,
  1.3348904870778067446e-20, -4.2254836195018827479e-37, 9.3011872068621332399e-54, 1.113250147552460308e-69,
  6.5726412927436632287e-21, 1.0820844071023395684e-36, 1.7634898158762432635e-52, 3.5887057810247033998e-68,
  3.1845095037264626247e-21, 3.2976802257607573031e-37, 9.3011872068621332399e-54, 1.113250147552460308e-69,
  1.4904436092178623228e-21, -4.6390169687056261795e-38, -1.1392999419355048437e-54, -4.587677453735884283e-71,
  6.4341066196356198368e-22, -4.6390169687056261795e-38, -1.1392999419355048437e-54, -4.587677453735884283e-71,
  2.1989418833641172011e-22, 4.7649378378726728402e-38, 9.3011872068621332399e-54, 1.113250147552460308e-69,
  8.135951522836682362e-24, 6.2960434583523738135e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73,
  8.135951522836682362e-24, 6.2960434583523738135e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73,
  8.135951522836682362e-24, 6.2960434583523738135e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73,
  8.135951522836682362e-24, 6.2960434583523738135e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73,
  8.135951522836682362e-24, 6.2960434583523738135e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73,
  1.5185066224124613304e-24, 2.6226236120327253511e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73,
  1.5185066224124613304e-24, 2.6226236120327253511e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73,
  1.5185066224124613304e-24, 2.6226236120327253511e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73,
  6.9132600985943383921e-25, 7.8591368887290111994e-41, 2.6283399642369020339e-57, 5.3358074162805516304e-73,
  2.7773570358292009361e-25, -1.3244127270701094468e-41, -2.4695541513869446866e-57, -3.2399200798614356002e-74,
  7.0940550444663151936e-26, 9.7147467687967058732e-42, 7.9392906424978921242e-59, 2.9745456030524896742e-75,
  7.0940550444663151936e-26, 9.7147467687967058732e-42, 7.9392906424978921242e-59, 2.9745456030524896742e-75,
  1.9241762160098927996e-26, 3.9750282589222551507e-42, 7.9392906424978921242e-59, 2.9745456030524896742e-75,
  1.9241762160098927996e-26, 3.9750282589222551507e-42, 7.9392906424978921242e-59, 2.9745456030524896742e-75,
  6.317065088957874881e-27, -3.2976062348358281152e-43, -2.6168913164368963837e-61, 3.7036201000008290615e-78,
  6.317065088957874881e-27, -3.2976062348358281152e-43, -2.6168913164368963837e-61, 3.7036201000008290615e-78,
  3.0858908211726098086e-27, 3.8770419025072344914e-43, 7.9392906424978921242e-59, 2.9745456030524896742e-75,
  1.4703036872799779898e-27, 2.8971783383570358633e-44, -2.6168913164368963837e-61, 3.7036201000008290615e-78,
  6.625101203336619011e-28, 2.8971783383570358633e-44, -2.6168913164368963837e-61, 3.7036201000008290615e-78,
  2.5861333686050385673e-28, 2.8971783383570358633e-44, -2.6168913164368963837e-61, 3.7036201000008290615e-78,
  5.6664945123924856962e-29, 6.5510079543732854985e-45, -2.6168913164368963837e-61, 3.7036201000008290615e-78,
  5.6664945123924856962e-29, 6.5510079543732854985e-45, -2.6168913164368963837e-61, 3.7036201000008290615e-78,
  6.1778471897801070206e-30, 9.4581409707401690366e-46, 4.9461632249367446986e-62, 3.7036201000008290615e-78,
  6.1778471897801070206e-30, 9.4581409707401690366e-46, 4.9461632249367446986e-62, 3.7036201000008290615e-78,
  6.1778471897801070206e-30, 9.4581409707401690366e-46, 4.9461632249367446986e-62, 3.7036201000008290615e-78,
  6.1778471897801070206e-30, 9.4581409707401690366e-46, 4.9461632249367446986e-62, 3.7036201000008290615e-78,
  3.0224035688960604996e-30, 2.451648649116083682e-46, 4.9461632249367446986e-62, 3.7036201000008290615e-78,
  1.4446817584540368888e-30, 2.451648649116083682e-46, 4.9461632249367446986e-62, 3.7036201000008290615e-78,
  6.5582085323302525856e-31, 7.0002556871006273225e-47, 1.0567786762735315635e-62, -6.1446417754639313137e-79,
  2.6139040062251944343e-31, -1.7578597149294783985e-47, 8.4432539107728090768e-64, 1.9517662449371102229e-79,
  6.4175174317266470186e-32, 4.3166913557804827486e-48, 8.4432539107728090768e-64, 1.9517662449371102229e-79,
  6.4175174317266470186e-32, 4.3166913557804827486e-48, 8.4432539107728090768e-64, 1.9517662449371102229e-79,
  1.4871367740953237822e-32, -1.1571307704883330232e-48, -6.7249112515659569668e-65, -7.2335760163150273591e-81,
  1.4871367740953237822e-32, -1.1571307704883330232e-48, -6.7249112515659569668e-65, -7.2335760163150273591e-81,
  2.5454160968749269937e-33, 2.1132476107887107169e-49, 8.7154294504188118783e-66, 1.2001823382693912203e-81,
  2.5454160968749269937e-33, 2.1132476107887107169e-49, 8.7154294504188118783e-66, 1.2001823382693912203e-81,
  2.5454160968749269937e-33, 2.1132476107887107169e-49, 8.7154294504188118783e-66, 1.2001823382693912203e-81,
  1.0046721413651383112e-33, 2.1132476107887107169e-49, 8.7154294504188118783e-66, 1.2001823382693912203e-81,
  2.3430016361024414106e-34, 4.0267819632970559834e-50, -7.8013829534098555144e-67, -1.1759240463442418271e-82,
  2.3430016361024414106e-34, 4.0267819632970559834e-50, -7.8013829534098555144e-67, -1.1759240463442418271e-82,
  4.1707169171520598517e-35, -2.4964157285045710972e-51, -1.866653112309982615e-67, 1.4185069655957361252e-83,
  4.1707169171520598517e-35, -2.4964157285045710972e-51, -1.866653112309982615e-67, 1.4185069655957361252e-83,
  4.1707169171520598517e-35, -2.4964157285045710972e-51, -1.866653112309982615e-67, 1.4185069655957361252e-83,
  1.7633044866680145008e-35, 2.8491136916798196016e-51, 4.0680767287898916022e-67, 1.4185069655957361252e-83,
  5.595982714259923599e-36, 1.7634898158762432635e-52, 3.588705781024702988e-68, 5.9489775128085140685e-84,
  5.595982714259923599e-36, 1.7634898158762432635e-52, 3.588705781024702988e-68, 5.9489775128085140685e-84,
  2.5867171761548675786e-36, 1.7634898158762432635e-52, 3.588705781024702988e-68, 5.9489775128085140685e-84,
  1.0820844071023395684e-36, 1.7634898158762432635e-52, 3.588705781024702988e-68, 5.9489775128085140685e-84,
  3.2976802257607573031e-37, 9.3011872068621332399e-54, 1.113250147552460308e-69, 2.9286284920280944778e-86,
  3.2976802257607573031e-37, 9.3011872068621332399e-54, 1.113250147552460308e-69, 2.9286284920280944778e-86,
  1.4168892644450972904e-37, 9.3011872068621332399e-54, 1.113250147552460308e-69, 2.9286284920280944778e-86,
  4.7649378378726728402e-38, 9.3011872068621332399e-54, 1.113250147552460308e-69, 2.9286284920280944778e-86,
  6.2960434583523738135e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.524218473063975309e-90,
  6.2960434583523738135e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.524218473063975309e-90,
  6.2960434583523738135e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.524218473063975309e-90,
  6.2960434583523738135e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.524218473063975309e-90,
  6.2960434583523738135e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.524218473063975309e-90,
  6.2960434583523738135e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.524218473063975309e-90,
  6.2960434583523738135e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.524218473063975309e-90,
  2.6226236120327253511e-40, 2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.524218473063975309e-90,
  7.8591368887290111994e-41, 2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.524218473063975309e-90,
  7.8591368887290111994e-41, 2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.524218473063975309e-90,
  3.2673620808294506214e-41, 2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.524218473063975309e-90,
  9.7147467687967058732e-42, 7.9392906424978921242e-59, 2.9745456030524891833e-75, 5.969437008257943935e-91,
  9.7147467687967058732e-42, 7.9392906424978921242e-59, 2.9745456030524891833e-75, 5.969437008257943935e-91,
  3.9750282589222551507e-42, 7.9392906424978921242e-59, 2.9745456030524891833e-75, 5.969437008257943935e-91,
  1.1051690039850297894e-42, 7.9392906424978921242e-59, 2.9745456030524891833e-75, 5.969437008257943935e-91,
  1.1051690039850297894e-42, 7.9392906424978921242e-59, 2.9745456030524891833e-75, 5.969437008257943935e-91,
  3.8770419025072344914e-43, 7.9392906424978921242e-59, 2.9745456030524891833e-75, 5.969437008257943935e-91,
  2.8971783383570358633e-44, -2.6168913164368963837e-61, 3.7036201000008285821e-78, 5.6554937751584084315e-94,
  2.8971783383570358633e-44, -2.6168913164368963837e-61, 3.7036201000008285821e-78, 5.6554937751584084315e-94,
  2.8971783383570358633e-44, -2.6168913164368963837e-61, 3.7036201000008285821e-78, 5.6554937751584084315e-94,
  2.8971783383570358633e-44, -2.6168913164368963837e-61, 3.7036201000008285821e-78, 5.6554937751584084315e-94,
  6.5510079543732854985e-45, -2.6168913164368963837e-61, 3.7036201000008285821e-78, 5.6554937751584084315e-94,
  6.5510079543732854985e-45, -2.6168913164368963837e-61, 3.7036201000008285821e-78, 5.6554937751584084315e-94,
  9.4581409707401690366e-46, 4.9461632249367446986e-62, 3.7036201000008285821e-78, 5.6554937751584084315e-94,
  9.4581409707401690366e-46, 4.9461632249367446986e-62, 3.7036201000008285821e-78, 5.6554937751584084315e-94,
  9.4581409707401690366e-46, 4.9461632249367446986e-62, 3.7036201000008285821e-78, 5.6554937751584084315e-94,
  2.451648649116083682e-46, 4.9461632249367446986e-62, 3.7036201000008285821e-78, 5.6554937751584084315e-94,
  2.451648649116083682e-46, 4.9461632249367446986e-62, 3.7036201000008285821e-78, 5.6554937751584084315e-94,
  7.0002556871006273225e-47, 1.0567786762735315635e-62, -6.1446417754639301152e-79, -1.5355611056488084652e-94,
  7.0002556871006273225e-47, 1.0567786762735315635e-62, -6.1446417754639301152e-79, -1.5355611056488084652e-94,
  2.6211979860855749482e-47, 8.4432539107728090768e-64, 1.9517662449371099233e-79, 2.62202614552995759e-95,
  4.3166913557804827486e-48, 8.4432539107728090768e-64, 1.9517662449371099233e-79, 2.62202614552995759e-95,
  4.3166913557804827486e-48, 8.4432539107728090768e-64, 1.9517662449371099233e-79, 2.62202614552995759e-95,
  4.3166913557804827486e-48, 8.4432539107728090768e-64, 1.9517662449371099233e-79, 2.62202614552995759e-95,
  1.5797802926460750146e-48, 2.3660905534865399025e-64, -7.2335760163150273591e-81, 2.8738690232659205689e-99,
  2.1132476107887107169e-49, 8.7154294504188118783e-66, 1.2001823382693912203e-81, 2.8738690232659205689e-99,
  2.1132476107887107169e-49, 8.7154294504188118783e-66, 1.2001823382693912203e-81, 2.8738690232659205689e-99,
  2.1132476107887107169e-49, 8.7154294504188118783e-66, 1.2001823382693912203e-81, 2.8738690232659205689e-99,
  4.0267819632970559834e-50, -7.8013829534098555144e-67, -1.1759240463442418271e-82, 2.8738690232659205689e-99,
  4.0267819632970559834e-50, -7.8013829534098555144e-67, -1.1759240463442418271e-82, 2.8738690232659205689e-99,
  4.0267819632970559834e-50, -7.8013829534098555144e-67, -1.1759240463442418271e-82, 2.8738690232659205689e-99,
  1.8885701952232994665e-50, -7.8013829534098555144e-67, -1.1759240463442418271e-82, 2.8738690232659205689e-99,
  8.1946431118642097069e-51, 1.5937536410989638719e-66, 1.459625439463388979e-82, 2.8738690232659205689e-99,
  2.8491136916798196016e-51, 4.0680767287898916022e-67, 1.4185069655957361252e-83, -7.8369062883735917115e-100,
  1.7634898158762432635e-52, 3.588705781024702988e-68, 5.9489775128085131541e-84, 1.0450891972142808004e-99,
  1.7634898158762432635e-52, 3.588705781024702988e-68, 5.9489775128085131541e-84, 1.0450891972142808004e-99,
  1.7634898158762432635e-52, 3.588705781024702988e-68, 5.9489775128085131541e-84, 1.0450891972142808004e-99,
  1.7634898158762432635e-52, 3.588705781024702988e-68, 5.9489775128085131541e-84, 1.0450891972142808004e-99,
  9.3011872068621332399e-54, 1.113250147552460308e-69, 2.9286284920280941206e-86, 2.1132026692048600853e-102,
  9.3011872068621332399e-54, 1.113250147552460308e-69, 2.9286284920280941206e-86, 2.1132026692048600853e-102,
  9.3011872068621332399e-54, 1.113250147552460308e-69, 2.9286284920280941206e-86, 2.1132026692048600853e-102,
  9.3011872068621332399e-54, 1.113250147552460308e-69, 2.9286284920280941206e-86, 2.1132026692048600853e-102,
  9.3011872068621332399e-54, 1.113250147552460308e-69, 2.9286284920280941206e-86, 2.1132026692048600853e-102,
  4.0809436324633147776e-54, -4.587677453735884283e-71, -2.8859500138942368532e-87, -5.6567402911297190423e-103,
  1.470821845263904967e-54, -4.587677453735884283e-71, -2.8859500138942368532e-87, -5.6567402911297190423e-103,
  1.6576095166419998917e-55, 2.6568658093254848067e-71, 5.1571087196495574384e-87, 3.2728487032630537605e-103,
  1.6576095166419998917e-55, 2.6568658093254848067e-71, 5.1571087196495574384e-87, 3.2728487032630537605e-103,
  1.6576095166419998917e-55, 2.6568658093254848067e-71, 5.1571087196495574384e-87, 3.2728487032630537605e-103,
  2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.5242184730639744369e-90, 1.145584788913072936e-105,
  2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.5242184730639744369e-90, 1.145584788913072936e-105,
  2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.5242184730639744369e-90, 1.145584788913072936e-105,
  2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.5242184730639744369e-90, 1.145584788913072936e-105,
  2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.5242184730639744369e-90, 1.145584788913072936e-105,
  2.6283399642369020339e-57, 5.3358074162805516304e-73, 4.5242184730639744369e-90, 1.145584788913072936e-105,
  7.9392906424978921242e-59, 2.9745456030524891833e-75, 5.969437008257942845e-91, 5.554706987098633963e-107,
  7.9392906424978921242e-59, 2.9745456030524891833e-75, 5.969437008257942845e-91, 5.554706987098633963e-107,
  7.9392906424978921242e-59, 2.9745456030524891833e-75, 5.969437008257942845e-91, 5.554706987098633963e-107,
  7.9392906424978921242e-59, 2.9745456030524891833e-75, 5.969437008257942845e-91, 5.554706987098633963e-107,
  7.9392906424978921242e-59, 2.9745456030524891833e-75, 5.969437008257942845e-91, 5.554706987098633963e-107,
  7.9392906424978921242e-59, 2.9745456030524891833e-75, 5.969437008257942845e-91, 5.554706987098633963e-107,
  3.9565608646667614317e-59, 2.9745456030524891833e-75, 5.969437008257942845e-91, 5.554706987098633963e-107,
  1.9651959757511960854e-59, 2.9745456030524891833e-75, 5.969437008257942845e-91, 5.554706987098633963e-107,
  9.6951353129341363331e-60, 7.6368645294831185015e-76, 1.0603435429602168369e-91, 1.0451839188820145747e-108,
  4.7167230906452229674e-60, 7.6368645294831185015e-76, 1.0603435429602168369e-91, 1.0451839188820145747e-108,
  2.2275169795007668372e-60, 2.1097166542226745549e-76, 4.4670685979800101779e-92, 1.0451839188820145747e-108,
  9.8291392392853877215e-61, -6.5385728340754726503e-77, -1.3520652573660833788e-93, -2.3220403312043059402e-109,
  3.6061239614242446325e-61, 7.2792968540756372162e-77, 1.3988851821689310822e-92, 1.0451839188820145747e-108,
  4.9461632249367446986e-62, 3.7036201000008285821e-78, 5.6554937751584084315e-94, -1.9306041120023063932e-110,
  4.9461632249367446986e-62, 3.7036201000008285821e-78, 5.6554937751584084315e-94, -1.9306041120023063932e-110,
  4.9461632249367446986e-62, 3.7036201000008285821e-78, 5.6554937751584084315e-94, -1.9306041120023063932e-110,
  1.0567786762735315635e-62, -6.1446417754639301152e-79, -1.535561105648808199e-94, -1.9306041120023063932e-110,
  1.0567786762735315635e-62, -6.1446417754639301152e-79, -1.535561105648808199e-94, -1.9306041120023063932e-110,
  8.4432539107728090768e-64, 1.9517662449371099233e-79, 2.62202614552995759e-95, 6.5314563001514358328e-112,
  8.4432539107728090768e-64, 1.9517662449371099233e-79, 2.62202614552995759e-95, 6.5314563001514358328e-112,
  8.4432539107728090768e-64, 1.9517662449371099233e-79, 2.62202614552995759e-95, 6.5314563001514358328e-112,
  8.4432539107728090768e-64, 1.9517662449371099233e-79, 2.62202614552995759e-95, 6.5314563001514358328e-112,
  2.3660905534865399025e-64, -7.2335760163150273591e-81, 2.8738690232659205689e-99, 1.8395411057335783574e-115,
  2.3660905534865399025e-64, -7.2335760163150273591e-81, 2.8738690232659205689e-99, 1.8395411057335783574e-115,
  8.4679971416497210292e-65, -7.2335760163150273591e-81, 2.8738690232659205689e-99, 1.8395411057335783574e-115,
  8.7154294504188118783e-66, 1.2001823382693912203e-81, 2.8738690232659205689e-99, 1.8395411057335783574e-115,
  8.7154294504188118783e-66, 1.2001823382693912203e-81, 2.8738690232659205689e-99, 1.8395411057335783574e-115,
  8.7154294504188118783e-66, 1.2001823382693912203e-81, 2.8738690232659205689e-99, 1.8395411057335783574e-115,
  8.7154294504188118783e-66, 1.2001823382693912203e-81, 2.8738690232659205689e-99, 1.8395411057335783574e-115,
  3.9676455775389135587e-66, 1.459625439463388979e-82, 2.8738690232659205689e-99, 1.8395411057335783574e-115,
  1.5937536410989638719e-66, 1.459625439463388979e-82, 2.8738690232659205689e-99, 1.8395411057335783574e-115,
  4.0680767287898916022e-67, 1.4185069655957361252e-83, -7.8369062883735917115e-100, -1.9081236411894110579e-116,
  4.0680767287898916022e-67, 1.4185069655957361252e-83, -7.8369062883735917115e-100, -1.9081236411894110579e-116,
  1.1007118082399544936e-67, 1.4185069655957361252e-83, -7.8369062883735917115e-100, -1.9081236411894110579e-116,
  1.1007118082399544936e-67, 1.4185069655957361252e-83, -7.8369062883735917115e-100, -1.9081236411894110579e-116,
  3.588705781024702988e-68, 5.9489775128085131541e-84, 1.0450891972142805974e-99, 1.8395411057335783574e-115,
  3.588705781024702988e-68, 5.9489775128085131541e-84, 1.0450891972142805974e-99, 1.8395411057335783574e-115,
  1.7341027056809927069e-68, 1.830931441234090934e-84, 1.3069928418846076386e-100, 3.1677600334418876704e-116,
  8.0680116800913756637e-69, -2.2809159455312046184e-85, -4.0748824503880445403e-101, -6.3915272253158644628e-117,
  3.4315039917320989315e-69, -2.2809159455312046184e-85, -4.0748824503880445403e-101, -6.3915272253158644628e-117,
  1.113250147552460308e-69, 2.9286284920280941206e-86, 2.1132026692048600853e-102, -4.6672632026740766185e-119,
  1.113250147552460308e-69, 2.9286284920280941206e-86, 2.1132026692048600853e-102, -4.6672632026740766185e-119,
  5.3368668650755071652e-70, 2.9286284920280941206e-86, 2.1132026692048600853e-102, -4.6672632026740766185e-119,
  2.4390495598509592076e-70, 2.9286284920280941206e-86, 2.1132026692048600853e-102, -4.6672632026740766185e-119,
  9.901409072386855505e-71, -2.8859500138942368532e-87, -5.6567402911297190423e-103, -4.6672632026740766185e-119,
  2.6568658093254848067e-71, 5.1571087196495574384e-87, 3.2728487032630532648e-103, 5.2465720993401781599e-119,
  2.6568658093254848067e-71, 5.1571087196495574384e-87, 3.2728487032630532648e-103, 5.2465720993401781599e-119,
  8.4572999356014273536e-72, 1.1355793528776598461e-87, 3.2728487032630532648e-103, 5.2465720993401781599e-119,
  8.4572999356014273536e-72, 1.1355793528776598461e-87, 3.2728487032630532648e-103, 5.2465720993401781599e-119,
  3.9294603961880721752e-72, 1.3019701118468578292e-88, -7.5747169634236195447e-105, -2.0152904854894729832e-121,
  1.6655406264813940833e-72, 1.3019701118468578292e-88, -7.5747169634236195447e-105, -2.0152904854894729832e-121,
  5.3358074162805516304e-73, 4.5242184730639744369e-90, 1.1455847889130727424e-105, 1.8573014293598455046e-121,
  5.3358074162805516304e-73, 4.5242184730639744369e-90, 1.1455847889130727424e-105, 1.8573014293598455046e-121,
  2.5059077041472040156e-73, 4.5242184730639744369e-90, 1.1455847889130727424e-105, 1.8573014293598455046e-121,
  1.0909578480805302081e-73, 4.5242184730639744369e-90, 1.1455847889130727424e-105, 1.8573014293598455046e-121,
  3.8348292004719330442e-74, 4.5242184730639744369e-90, 1.1455847889130727424e-105, 1.8573014293598455046e-121,
  2.9745456030524891833e-75, 5.969437008257942845e-91, 5.5547069870986327528e-107, 1.6304246661326865276e-122,
  2.9745456030524891833e-75, 5.969437008257942845e-91, 5.5547069870986327528e-107, 1.6304246661326865276e-122,
  2.9745456030524891833e-75, 5.969437008257942845e-91, 5.5547069870986327528e-107, 1.6304246661326865276e-122,
  2.9745456030524891833e-75, 5.969437008257942845e-91, 5.5547069870986327528e-107, 1.6304246661326865276e-122,
  7.6368645294831185015e-76, 1.0603435429602168369e-91, 1.0451839188820145747e-108, 4.2386081393205242443e-125,
  7.6368645294831185015e-76, 1.0603435429602168369e-91, 1.0451839188820145747e-108, 4.2386081393205242443e-125,
  2.1097166542226745549e-76, 4.4670685979800101779e-92, 1.0451839188820145747e-108, 4.2386081393205242443e-125,
  2.1097166542226745549e-76, 4.4670685979800101779e-92, 1.0451839188820145747e-108, 4.2386081393205242443e-125,
  7.2792968540756372162e-77, 1.3988851821689310822e-92, 1.0451839188820145747e-108, 4.2386081393205242443e-125,
  3.7036201000008285821e-78, 5.6554937751584084315e-94, -1.9306041120023063932e-110, 1.0223371855251472933e-126,
  3.7036201000008285821e-78, 5.6554937751584084315e-94, -1.9306041120023063932e-110, 1.0223371855251472933e-126,
  3.7036201000008285821e-78, 5.6554937751584084315e-94, -1.9306041120023063932e-110, 1.0223371855251472933e-126,
  3.7036201000008285821e-78, 5.6554937751584084315e-94, -1.9306041120023063932e-110, 1.0223371855251472933e-126,
  3.7036201000008285821e-78, 5.6554937751584084315e-94, -1.9306041120023063932e-110, 1.0223371855251472933e-126,
  1.5445779612272179051e-78, 8.6145718795359707834e-95, 7.3062078800278780675e-111, 1.0223371855251472933e-126,
  4.6505689184041232695e-79, 8.6145718795359707834e-95, 7.3062078800278780675e-111, 1.0223371855251472933e-126,
  4.6505689184041232695e-79, 8.6145718795359707834e-95, 7.3062078800278780675e-111, 1.0223371855251472933e-126,
  1.9517662449371099233e-79, 2.62202614552995759e-95, 6.5314563001514349095e-112, 9.9039323746573674262e-128,
  6.0236490820360325022e-80, -3.7424672147304925625e-96, -1.784871512364483542e-112, 6.7095375687163151728e-129,
  6.0236490820360325022e-80, -3.7424672147304925625e-96, -1.784871512364483542e-112, 6.7095375687163151728e-129,
  2.6501457402022643213e-80, 3.7482149527770239293e-96, 6.5314563001514349095e-112, 9.9039323746573674262e-128,
  9.6339406928538097998e-81, 2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132,
  1.2001823382693912203e-81, 2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132,
  1.2001823382693912203e-81, 2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132,
  1.2001823382693912203e-81, 2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132,
  1.459625439463388979e-82, 2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132,
  1.459625439463388979e-82, 2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132,
  1.459625439463388979e-82, 2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132,
  1.4185069655957361252e-83, -7.8369062883735917115e-100, -1.9081236411894107761e-116, -2.1796760241698337334e-132,
  1.4185069655957361252e-83, -7.8369062883735917115e-100, -1.9081236411894107761e-116, -2.1796760241698337334e-132,
  1.4185069655957361252e-83, -7.8369062883735917115e-100, -1.9081236411894107761e-116, -2.1796760241698337334e-132,
  1.4185069655957361252e-83, -7.8369062883735917115e-100, -1.9081236411894107761e-116, -2.1796760241698337334e-132,
  5.9489775128085131541e-84, 1.0450891972142805974e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132,
  1.830931441234090934e-84, 1.3069928418846076386e-100, 3.1677600334418871069e-116, 3.4556869017247800778e-132,
  1.830931441234090934e-84, 1.3069928418846076386e-100, 3.1677600334418871069e-116, 3.4556869017247800778e-132,
  8.0141992334048515034e-85, 1.3069928418846076386e-100, 3.1677600334418871069e-116, 3.4556869017247800778e-132,
  2.8666416439368237283e-85, 1.6400545060233297363e-101, -4.6672632026740766185e-119, -3.755176715260116501e-136,
  2.9286284920280941206e-86, 2.1132026692048600853e-102, -4.6672632026740766185e-119, -3.755176715260116501e-136,
  2.9286284920280941206e-86, 2.1132026692048600853e-102, -4.6672632026740766185e-119, -3.755176715260116501e-136,
  2.9286284920280941206e-86, 2.1132026692048600853e-102, -4.6672632026740766185e-119, -3.755176715260116501e-136,
  2.9286284920280941206e-86, 2.1132026692048600853e-102, -4.6672632026740766185e-119, -3.755176715260116501e-136,
  1.3200167453193350837e-86, 2.1132026692048600853e-102, -4.6672632026740766185e-119, -3.755176715260116501e-136,
  5.1571087196495574384e-87, 3.2728487032630532648e-103, 5.2465720993401781599e-119, -3.755176715260116501e-136,
  1.1355793528776598461e-87, 3.2728487032630532648e-103, 5.2465720993401781599e-119, -3.755176715260116501e-136,
  1.1355793528776598461e-87, 3.2728487032630532648e-103, 5.2465720993401781599e-119, -3.755176715260116501e-136,
  1.3019701118468578292e-88, -7.5747169634236195447e-105, -2.0152904854894725532e-121, -3.1562414818576682143e-137,
  1.3019701118468578292e-88, -7.5747169634236195447e-105, -2.0152904854894725532e-121, -3.1562414818576682143e-137,
  1.3019701118468578292e-88, -7.5747169634236195447e-105, -2.0152904854894725532e-121, -3.1562414818576682143e-137,
  4.5242184730639744369e-90, 1.1455847889130727424e-105, 1.8573014293598452896e-121, 1.1431992269852683481e-137,
  4.5242184730639744369e-90, 1.1455847889130727424e-105, 1.8573014293598452896e-121, 1.1431992269852683481e-137,
  4.5242184730639744369e-90, 1.1455847889130727424e-105, 1.8573014293598452896e-121, 1.1431992269852683481e-137,
  4.5242184730639744369e-90, 1.1455847889130727424e-105, 1.8573014293598452896e-121, 1.1431992269852683481e-137,
  4.5242184730639744369e-90, 1.1455847889130727424e-105, 1.8573014293598452896e-121, 1.1431992269852683481e-137,
  5.969437008257942845e-91, 5.5547069870986327528e-107, 1.6304246661326865276e-122, 6.8339049774534162772e-139,
  5.969437008257942845e-91, 5.5547069870986327528e-107, 1.6304246661326865276e-122, 6.8339049774534162772e-139,
  5.969437008257942845e-91, 5.5547069870986327528e-107, 1.6304246661326865276e-122, 6.8339049774534162772e-139,
  1.0603435429602168369e-91, 1.0451839188820145747e-108, 4.2386081393205242443e-125, 1.1062055705591188256e-141,
  1.0603435429602168369e-91, 1.0451839188820145747e-108, 4.2386081393205242443e-125, 1.1062055705591188256e-141,
  1.0603435429602168369e-91, 1.0451839188820145747e-108, 4.2386081393205242443e-125, 1.1062055705591188256e-141,
  4.4670685979800101779e-92, 1.0451839188820145747e-108, 4.2386081393205242443e-125, 1.1062055705591188256e-141,
  1.3988851821689310822e-92, 1.0451839188820145747e-108, 4.2386081393205242443e-125, 1.1062055705591188256e-141,
  1.3988851821689310822e-92, 1.0451839188820145747e-108, 4.2386081393205242443e-125, 1.1062055705591188256e-141,
  6.3183932821616130831e-93, 1.0451839188820145747e-108, 4.2386081393205242443e-125, 1.1062055705591188256e-141,
  2.4831640123977650651e-93, 1.9359195088038447797e-109, -4.8867691298577234423e-126, -2.0587960670007823264e-142,
  5.6554937751584084315e-94, -1.9306041120023063932e-110, 1.0223371855251471293e-126, 1.2214168761472102282e-142,
  5.6554937751584084315e-94, -1.9306041120023063932e-110, 1.0223371855251471293e-126, 1.2214168761472102282e-142,
  8.6145718795359707834e-95, 7.3062078800278780675e-111, 1.0223371855251471293e-126, 1.2214168761472102282e-142,
  8.6145718795359707834e-95, 7.3062078800278780675e-111, 1.0223371855251471293e-126, 1.2214168761472102282e-142,
  8.6145718795359707834e-95, 7.3062078800278780675e-111, 1.0223371855251471293e-126, 1.2214168761472102282e-142,
  2.62202614552995759e-95, 6.5314563001514349095e-112, 9.9039323746573674262e-128, -8.6629775332868987041e-145,
  2.62202614552995759e-95, 6.5314563001514349095e-112, 9.9039323746573674262e-128, -8.6629775332868987041e-145,
  1.1238897120284541253e-95, 6.5314563001514349095e-112, 9.9039323746573674262e-128, -8.6629775332868987041e-145,
  3.7482149527770239293e-96, 6.5314563001514349095e-112, 9.9039323746573674262e-128, -8.6629775332868987041e-145,
  2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148,
  2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148,
  2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148,
  2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148,
  2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148,
  2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148,
  2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148,
  2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148,
  2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148,
  2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148,
  2.8738690232659205689e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148,
  1.0450891972142805974e-99, 1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148,
  1.3069928418846076386e-100, 3.1677600334418871069e-116, 3.4556869017247794521e-132, 8.5448727249069983612e-148,
  1.3069928418846076386e-100, 3.1677600334418871069e-116, 3.4556869017247794521e-132, 8.5448727249069983612e-148,
  1.3069928418846076386e-100, 3.1677600334418871069e-116, 3.4556869017247794521e-132, 8.5448727249069983612e-148,
  1.6400545060233297363e-101, -4.6672632026740766185e-119, -3.755176715260116501e-136, 2.1571619860435652883e-152,
  1.6400545060233297363e-101, -4.6672632026740766185e-119, -3.755176715260116501e-136, 2.1571619860435652883e-152,
  1.6400545060233297363e-101, -4.6672632026740766185e-119, -3.755176715260116501e-136, 2.1571619860435652883e-152,
  2.1132026692048600853e-102, -4.6672632026740766185e-119, -3.755176715260116501e-136, 2.1571619860435652883e-152,
  2.1132026692048600853e-102, -4.6672632026740766185e-119, -3.755176715260116501e-136, 2.1571619860435652883e-152,
  2.1132026692048600853e-102, -4.6672632026740766185e-119, -3.755176715260116501e-136, 2.1571619860435652883e-152,
  3.2728487032630532648e-103, 5.2465720993401781599e-119, -3.755176715260116501e-136, 2.1571619860435652883e-152,
  3.2728487032630532648e-103, 5.2465720993401781599e-119, -3.755176715260116501e-136, 2.1571619860435652883e-152,
  3.2728487032630532648e-103, 5.2465720993401781599e-119, -3.755176715260116501e-136, 2.1571619860435652883e-152,
  1.0404514546648604359e-103, 2.896544483330507019e-120, 3.1239284188885823808e-136, 2.1571619860435652883e-152,
  1.0404514546648604359e-103, 2.896544483330507019e-120, 3.1239284188885823808e-136, 2.1571619860435652883e-152,
  4.8235214251531210473e-104, 2.896544483330507019e-120, 3.1239284188885823808e-136, 2.1571619860435652883e-152,
  2.0330248644053793915e-104, 2.896544483330507019e-120, 3.1239284188885823808e-136, 2.1571619860435652883e-152,
  6.3777658403150887343e-105, -2.0152904854894725532e-121, -3.156241481857667737e-137, -7.0684085473731388916e-153,
  6.3777658403150887343e-105, -2.0152904854894725532e-121, -3.156241481857667737e-137, -7.0684085473731388916e-153,
  2.88964513938041089e-105, 5.7298933442091639924e-121, -3.156241481857667737e-137, -7.0684085473731388916e-153,
  1.1455847889130727424e-105, 1.8573014293598452896e-121, 1.1431992269852681095e-137, 2.4782675885631257398e-153,
  2.7355461367940366859e-106, -7.8994528064813712419e-123, -2.0037599452814940222e-138, 9.1598554579059548847e-155,
  2.7355461367940366859e-106, -7.8994528064813712419e-123, -2.0037599452814940222e-138, 9.1598554579059548847e-155,
  5.5547069870986327528e-107, 1.6304246661326865276e-122, 6.8339049774534147855e-139, 9.1598554579059548847e-155,
  5.5547069870986327528e-107, 1.6304246661326865276e-122, 6.8339049774534147855e-139, 9.1598554579059548847e-155,
  1.0451839188820145747e-108, 4.2386081393205242443e-125, 1.1062055705591186799e-141, 1.1734404793201255869e-157,
  1.0451839188820145747e-108, 4.2386081393205242443e-125, 1.1062055705591186799e-141, 1.1734404793201255869e-157,
  1.0451839188820145747e-108, 4.2386081393205242443e-125, 1.1062055705591186799e-141, 1.1734404793201255869e-157,
  1.0451839188820145747e-108, 4.2386081393205242443e-125, 1.1062055705591186799e-141, 1.1734404793201255869e-157,
  1.0451839188820145747e-108, 4.2386081393205242443e-125, 1.1062055705591186799e-141, 1.1734404793201255869e-157,
  1.0451839188820145747e-108, 4.2386081393205242443e-125, 1.1062055705591186799e-141, 1.1734404793201255869e-157,
  1.9359195088038447797e-109, -4.8867691298577234423e-126, -2.0587960670007819622e-142, -2.8326669474241479263e-158,
  1.9359195088038447797e-109, -4.8867691298577234423e-126, -2.0587960670007819622e-142, -2.8326669474241479263e-158,
  1.9359195088038447797e-109, -4.8867691298577234423e-126, -2.0587960670007819622e-142, -2.8326669474241479263e-158,
  8.7142954880180709975e-110, -4.8867691298577234423e-126, -2.0587960670007819622e-142, -2.8326669474241479263e-158,
  3.3918456880078814158e-110, 6.931443500908017045e-126, 1.1062055705591186799e-141, 1.1734404793201255869e-157,
  7.3062078800278780675e-111, 1.0223371855251471293e-126, 1.2214168761472102282e-142, 8.0910098773220312367e-159,
  7.3062078800278780675e-111, 1.0223371855251471293e-126, 1.2214168761472102282e-142, 8.0910098773220312367e-159,
  6.5314563001514349095e-112, 9.9039323746573674262e-128, -8.6629775332868972816e-145, -1.5987060076657616072e-160,
  6.5314563001514349095e-112, 9.9039323746573674262e-128, -8.6629775332868972816e-145, -1.5987060076657616072e-160,
  6.5314563001514349095e-112, 9.9039323746573674262e-128, -8.6629775332868972816e-145, -1.5987060076657616072e-160,
  6.5314563001514349095e-112, 9.9039323746573674262e-128, -8.6629775332868972816e-145, -1.5987060076657616072e-160,
  2.3732923938934761454e-112, 6.7095375687163138915e-129, 1.6963686085056791706e-144, 1.2464251916751375716e-160,
  2.9421044076449630171e-113, 6.7095375687163138915e-129, 1.6963686085056791706e-144, 1.2464251916751375716e-160,
  2.9421044076449630171e-113, 6.7095375687163138915e-129, 1.6963686085056791706e-144, 1.2464251916751375716e-160,
  2.9421044076449630171e-113, 6.7095375687163138915e-129, 1.6963686085056791706e-144, 1.2464251916751375716e-160,
  3.4325196623373878948e-114, 9.3892593260023063019e-130, 9.4702132359198537748e-146, 1.7950099192230045857e-161,
  3.4325196623373878948e-114, 9.3892593260023063019e-130, 9.4702132359198537748e-146, 1.7950099192230045857e-161,
  3.4325196623373878948e-114, 9.3892593260023063019e-130, 9.4702132359198537748e-146, 1.7950099192230045857e-161,
  1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148, 2.9106774506606945839e-164,
  1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148, 2.9106774506606945839e-164,
  1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148, 2.9106774506606945839e-164,
  1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148, 2.9106774506606945839e-164,
  1.8395411057335783574e-115, -7.8150389500644475446e-132, -3.9681466199873824165e-148, 2.9106774506606945839e-164,
  8.2436437080731844263e-116, 1.4726412753514008951e-131, -3.9681466199873824165e-148, 2.9106774506606945839e-164,
  3.1677600334418871069e-116, 3.4556869017247794521e-132, 8.544872724906996972e-148, 1.6802919634942429241e-163,
  6.2981819612623816536e-117, 6.3800543877747317218e-133, 7.2423563434801054878e-149, 1.1741471776254779927e-164,
  6.2981819612623816536e-117, 6.3800543877747317218e-133, 7.2423563434801054878e-149, 1.1741471776254779927e-164,
  6.2981819612623816536e-117, 6.3800543877747317218e-133, 7.2423563434801054878e-149, 1.1741471776254779927e-164,
  3.1257546646178208289e-117, -6.6414926959353515111e-134, -5.7828074707888119584e-150, -1.2825052715093464343e-165,
  1.5395410162955400644e-117, -6.6414926959353515111e-134, -5.7828074707888119584e-150, -1.2825052715093464343e-165,
  7.4643419213439950602e-118, 1.0969016447485317626e-133, -5.7828074707888119584e-150, -1.2825052715093464343e-165,
  3.4988078005382940294e-118, 2.1637618757749825688e-134, -8.9490928918944555247e-151, -1.9717385086233606481e-166,
  1.5160407401354430737e-118, 2.1637618757749825688e-134, -8.9490928918944555247e-151, -1.9717385086233606481e-166,
  5.2465720993401781599e-119, -3.755176715260116501e-136, 2.1571619860435648643e-152, 6.3257905089784152346e-168,
  2.896544483330507019e-120, 3.1239284188885823808e-136, 2.1571619860435648643e-152, 6.3257905089784152346e-168,
  2.896544483330507019e-120, 3.1239284188885823808e-136, 2.1571619860435648643e-152, 6.3257905089784152346e-168,
  2.896544483330507019e-120, 3.1239284188885823808e-136, 2.1571619860435648643e-152, 6.3257905089784152346e-168,
  2.896544483330507019e-120, 3.1239284188885823808e-136, 2.1571619860435648643e-152, 6.3257905089784152346e-168,
  2.896544483330507019e-120, 3.1239284188885823808e-136, 2.1571619860435648643e-152, 6.3257905089784152346e-168,
  1.3475077173907800538e-120, -3.156241481857667737e-137, -7.0684085473731388916e-153, -3.3573283875161501977e-170,
  5.7298933442091639924e-121, -3.156241481857667737e-137, -7.0684085473731388916e-153, -3.3573283875161501977e-170,
  1.8573014293598452896e-121, 1.1431992269852681095e-137, 2.4782675885631257398e-153, -3.3573283875161501977e-170,
  1.8573014293598452896e-121, 1.1431992269852681095e-137, 2.4782675885631257398e-153, -3.3573283875161501977e-170,
  8.8915345064751572143e-122, 1.1431992269852681095e-137, 2.4782675885631257398e-153, -3.3573283875161501977e-170,
  4.0507946129135104481e-122, 6.8339049774534147855e-139, 9.1598554579059548847e-155, -4.5159745404911825673e-172,
  1.6304246661326865276e-122, 6.8339049774534147855e-139, 9.1598554579059548847e-155, -4.5159745404911825673e-172,
  4.2023969274227456735e-123, 6.8339049774534147855e-139, 9.1598554579059548847e-155, -4.5159745404911825673e-172,
  4.2023969274227456735e-123, 6.8339049774534147855e-139, 9.1598554579059548847e-155, -4.5159745404911825673e-172,
  1.1769344939467164447e-123, 1.1602886988632691941e-140, 3.0307583960570927356e-156, 5.8345524661064369683e-172,
  1.1769344939467164447e-123, 1.1602886988632691941e-140, 3.0307583960570927356e-156, 5.8345524661064369683e-172,
  4.2056888557770896953e-124, 1.1602886988632691941e-140, 3.0307583960570927356e-156, 5.8345524661064369683e-172,
  4.2386081393205242443e-125, 1.1062055705591186799e-141, 1.1734404793201255869e-157, 1.2381024895275844856e-174,
  4.2386081393205242443e-125, 1.1062055705591186799e-141, 1.1734404793201255869e-157, 1.2381024895275844856e-174,
  4.2386081393205242443e-125, 1.1062055705591186799e-141, 1.1734404793201255869e-157, 1.2381024895275844856e-174,
  4.2386081393205242443e-125, 1.1062055705591186799e-141, 1.1734404793201255869e-157, 1.2381024895275844856e-174,
  1.8749656131673758844e-125, 1.1062055705591186799e-141, 1.1734404793201255869e-157, 1.2381024895275844856e-174,
  6.931443500908017045e-126, 1.1062055705591186799e-141, 1.1734404793201255869e-157, 1.2381024895275844856e-174,
  1.0223371855251471293e-126, 1.2214168761472102282e-142, 8.0910098773220302259e-159, 1.2381024895275844856e-174,
  1.0223371855251471293e-126, 1.2214168761472102282e-142, 8.0910098773220302259e-159, 1.2381024895275844856e-174,
  1.0223371855251471293e-126, 1.2214168761472102282e-142, 8.0910098773220302259e-159, 1.2381024895275844856e-174,
  2.8369889610228834887e-127, 4.0136364036021218058e-143, -1.0134099605688458828e-159, -2.5389576707476506925e-176,
  2.8369889610228834887e-127, 4.0136364036021218058e-143, -1.0134099605688458828e-159, -2.5389576707476506925e-176,
  9.9039323746573674262e-128, -8.6629775332868972816e-145, -1.5987060076657612913e-160, -2.5389576707476506925e-176,
  6.7095375687163138915e-129, 1.6963686085056791706e-144, 1.2464251916751375716e-160, 6.197724948400014906e-177,
  6.7095375687163138915e-129, 1.6963686085056791706e-144, 1.2464251916751375716e-160, 6.197724948400014906e-177,
  6.7095375687163138915e-129, 1.6963686085056791706e-144, 1.2464251916751375716e-160, 6.197724948400014906e-177,
  6.7095375687163138915e-129, 1.6963686085056791706e-144, 1.2464251916751375716e-160, 6.197724948400014906e-177,
  9.3892593260023063019e-130, 9.4702132359198537748e-146, 1.7950099192230045857e-161, -1.6991004655691155518e-177,
  9.3892593260023063019e-130, 9.4702132359198537748e-146, 1.7950099192230045857e-161, -1.6991004655691155518e-177,
  9.3892593260023063019e-130, 9.4702132359198537748e-146, 1.7950099192230045857e-161, -1.6991004655691155518e-177,
  2.175994780857201024e-130, 1.4618808551874518553e-146, 1.6802919634942426156e-163, 2.8330093736631818036e-179,
  2.175994780857201024e-130, 1.4618808551874518553e-146, 1.6802919634942426156e-163, 2.8330093736631818036e-179,
  3.7267864457092460442e-131, 4.6083930759590139305e-147, 1.6802919634942426156e-163, 2.8330093736631818036e-179,
  3.7267864457092460442e-131, 4.6083930759590139305e-147, 1.6802919634942426156e-163, 2.8330093736631818036e-179,
  3.7267864457092460442e-131, 4.6083930759590139305e-147, 1.6802919634942426156e-163, 2.8330093736631818036e-179,
  1.4726412753514008951e-131, -3.9681466199873824165e-148, 2.9106774506606941983e-164, 5.1948630316441296498e-180,
  3.4556869017247794521e-132, 8.544872724906996972e-148, 1.6802919634942426156e-163, 2.8330093736631818036e-179,
  3.4556869017247794521e-132, 8.544872724906996972e-148, 1.6802919634942426156e-163, 2.8330093736631818036e-179,
  6.3800543877747317218e-133, 7.2423563434801054878e-149, 1.1741471776254777999e-164, 1.3389912474795152755e-180,
  6.3800543877747317218e-133, 7.2423563434801054878e-149, 1.1741471776254777999e-164, 1.3389912474795152755e-180,
  6.3800543877747317218e-133, 7.2423563434801054878e-149, 1.1741471776254777999e-164, 1.3389912474795152755e-180,
  2.8579525590905986764e-133, -5.7828074707888119584e-150, -1.2825052715093464343e-165, -1.0696067158221530218e-181,
  1.0969016447485317626e-133, -5.7828074707888119584e-150, -1.2825052715093464343e-165, -1.0696067158221530218e-181,
  2.1637618757749825688e-134, -8.9490928918944555247e-151, -1.9717385086233606481e-166, 1.3535321672928907047e-182,
  2.1637618757749825688e-134, -8.9490928918944555247e-151, -1.9717385086233606481e-166, 1.3535321672928907047e-182,
  2.1637618757749825688e-134, -8.9490928918944555247e-151, -1.9717385086233606481e-166, 1.3535321672928907047e-182,
  1.0631050543111905033e-134, 1.5490398016102376505e-150, 3.4549185946116918017e-166, 1.3535321672928907047e-182,
  5.1277664357929471499e-135, 3.2706525621039604902e-151, 7.4159004299416557678e-167, 1.3535321672928907047e-182,
  2.3761243821334675971e-135, 3.2706525621039604902e-151, 7.4159004299416557678e-167, 1.3535321672928907047e-182,
  1.0003033553037281263e-135, 2.1571619860435648643e-152, 6.3257905089784152346e-168, 3.5607241064750984115e-184,
  3.1239284188885823808e-136, 2.1571619860435648643e-152, 6.3257905089784152346e-168, 3.5607241064750984115e-184,
  3.1239284188885823808e-136, 2.1571619860435648643e-152, 6.3257905089784152346e-168, 3.5607241064750984115e-184,
  1.4041521353514076604e-136, 2.1571619860435648643e-152, 6.3257905089784152346e-168, 3.5607241064750984115e-184,
  5.4426399358282049106e-137, 2.4782675885631257398e-153, -3.3573283875161501977e-170, 3.0568054078295488291e-186,
  1.1431992269852681095e-137, 2.4782675885631257398e-153, -3.3573283875161501977e-170, 3.0568054078295488291e-186,
  1.1431992269852681095e-137, 2.4782675885631257398e-153, -3.3573283875161501977e-170, 3.0568054078295488291e-186,
  6.8339049774534147855e-139, 9.1598554579059548847e-155, -4.5159745404911819927e-172, -4.5870810097328578981e-188,
  6.8339049774534147855e-139, 9.1598554579059548847e-155, -4.5159745404911819927e-172, -4.5870810097328578981e-188,
  6.8339049774534147855e-139, 9.1598554579059548847e-155, -4.5159745404911819927e-172, -4.5870810097328578981e-188,
  6.8339049774534147855e-139, 9.1598554579059548847e-155, -4.5159745404911819927e-172, -4.5870810097328578981e-188,
  1.1602886988632691941e-140, 3.0307583960570927356e-156, 5.8345524661064358191e-172, 6.9043123899963188689e-188,
  1.1602886988632691941e-140, 3.0307583960570927356e-156, 5.8345524661064358191e-172, 6.9043123899963188689e-188,
  1.1602886988632691941e-140, 3.0307583960570927356e-156, 5.8345524661064358191e-172, 6.9043123899963188689e-188,
  1.1602886988632691941e-140, 3.0307583960570927356e-156, 5.8345524661064358191e-172, 6.9043123899963188689e-188,
  1.1602886988632691941e-140, 3.0307583960570927356e-156, 5.8345524661064358191e-172, 6.9043123899963188689e-188,
  1.1602886988632691941e-140, 3.0307583960570927356e-156, 5.8345524661064358191e-172, 6.9043123899963188689e-188,
  1.1062055705591186799e-141, 1.1734404793201255869e-157, 1.2381024895275844856e-174, -8.4789520282639751913e-191,
  1.1062055705591186799e-141, 1.1734404793201255869e-157, 1.2381024895275844856e-174, -8.4789520282639751913e-191,
  1.1062055705591186799e-141, 1.1734404793201255869e-157, 1.2381024895275844856e-174, -8.4789520282639751913e-191,
  1.1062055705591186799e-141, 1.1734404793201255869e-157, 1.2381024895275844856e-174, -8.4789520282639751913e-191,
  4.5016298192952031469e-142, -2.8326669474241479263e-158, 1.2381024895275844856e-174, -8.4789520282639751913e-191,
  1.2214168761472102282e-142, 8.0910098773220302259e-159, 1.2381024895275844856e-174, -8.4789520282639751913e-191,
  1.2214168761472102282e-142, 8.0910098773220302259e-159, 1.2381024895275844856e-174, -8.4789520282639751913e-191,
  4.0136364036021218058e-143, -1.0134099605688458828e-159, -2.5389576707476506925e-176, -6.2404128071707654958e-193,
  4.0136364036021218058e-143, -1.0134099605688458828e-159, -2.5389576707476506925e-176, -6.2404128071707654958e-193,
  1.9635033141346264592e-143, -1.0134099605688458828e-159, -2.5389576707476506925e-176, -6.2404128071707654958e-193,
  9.3843676940087855824e-144, 1.2626949989038732076e-159, 2.2730883653953564668e-175, 2.7431118386590483722e-191,
  4.2590349703400483539e-144, 1.2464251916751375716e-160, 6.1977249484000140293e-177, 1.1294061984896458822e-192,
  1.6963686085056791706e-144, 1.2464251916751375716e-160, 6.1977249484000140293e-177, 1.1294061984896458822e-192,
  4.1503542758849472122e-145, -1.7614040799531193879e-161, -1.6991004655691153326e-177, -1.856794109153959173e-193,
  4.1503542758849472122e-145, -1.7614040799531193879e-161, -1.6991004655691153326e-177, -1.856794109153959173e-193,
  9.4702132359198537748e-146, 1.7950099192230045857e-161, -1.6991004655691153326e-177, -1.856794109153959173e-193,
  9.4702132359198537748e-146, 1.7950099192230045857e-161, -1.6991004655691153326e-177, -1.856794109153959173e-193,
  1.4618808551874518553e-146, 1.6802919634942426156e-163, 2.8330093736631818036e-179, -7.4549709281190454638e-196,
  1.4618808551874518553e-146, 1.6802919634942426156e-163, 2.8330093736631818036e-179, -7.4549709281190454638e-196,
  1.4618808551874518553e-146, 1.6802919634942426156e-163, 2.8330093736631818036e-179, -7.4549709281190454638e-196,
  4.6083930759590139305e-147, 1.6802919634942426156e-163, 2.8330093736631818036e-179, -7.4549709281190454638e-196,
  4.6083930759590139305e-147, 1.6802919634942426156e-163, 2.8330093736631818036e-179, -7.4549709281190454638e-196,
  2.105789206980137775e-147, 1.6802919634942426156e-163, 2.8330093736631818036e-179, -7.4549709281190454638e-196,
  8.544872724906996972e-148, 1.6802919634942426156e-163, 2.8330093736631818036e-179, -7.4549709281190454638e-196,
  2.2883630524598079723e-148, 2.9106774506606941983e-164, 5.1948630316441287936e-180, 9.6685396110091032843e-196,
  2.2883630524598079723e-148, 2.9106774506606941983e-164, 5.1948630316441287936e-180, 9.6685396110091032843e-196,
  7.2423563434801054878e-149, 1.1741471776254777999e-164, 1.3389912474795150614e-180, 1.1067843414450286726e-196,
  7.2423563434801054878e-149, 1.1741471776254777999e-164, 1.3389912474795150614e-180, 1.1067843414450286726e-196,
  3.3320377982006123631e-149, 3.0588204110786950436e-165, 3.7502330143836152136e-181, 3.6564932749519464998e-198,
  1.3768785255608653665e-149, 3.0588204110786950436e-165, 3.7502330143836152136e-181, 3.6564932749519464998e-198,
  3.9929888924099219388e-150, -1.9717385086233606481e-166, 1.3535321672928907047e-182, 3.1205762277848031878e-199,
  3.9929888924099219388e-150, -1.9717385086233606481e-166, 1.3535321672928907047e-182, 3.1205762277848031878e-199,
  1.5490398016102376505e-150, 3.4549185946116918017e-166, 1.3535321672928907047e-182, 3.1205762277848031878e-199,
  3.2706525621039604902e-151, 7.4159004299416557678e-167, 1.3535321672928907047e-182, 3.1205762277848031878e-199,
  3.2706525621039604902e-151, 7.4159004299416557678e-167, 1.3535321672928907047e-182, 3.1205762277848031878e-199,
  2.1571619860435648643e-152, 6.3257905089784152346e-168, 3.5607241064750984115e-184, -1.4832196127821708615e-201,
  2.1571619860435648643e-152, 6.3257905089784152346e-168, 3.5607241064750984115e-184, -1.4832196127821708615e-201,
  2.1571619860435648643e-152, 6.3257905089784152346e-168, 3.5607241064750984115e-184, -1.4832196127821708615e-201,
  2.1571619860435648643e-152, 6.3257905089784152346e-168, 3.5607241064750984115e-184, -1.4832196127821708615e-201,
  2.4782675885631257398e-153, -3.3573283875161501977e-170, 3.0568054078295488291e-186, 1.4980560800565462618e-202,
  2.4782675885631257398e-153, -3.3573283875161501977e-170, 3.0568054078295488291e-186, 1.4980560800565462618e-202,
  2.4782675885631257398e-153, -3.3573283875161501977e-170, 3.0568054078295488291e-186, 1.4980560800565462618e-202,
  9.1598554579059548847e-155, -4.5159745404911819927e-172, -4.5870810097328572602e-188, -3.2905064432040069127e-204,
  9.1598554579059548847e-155, -4.5159745404911819927e-172, -4.5870810097328572602e-188, -3.2905064432040069127e-204,
  9.1598554579059548847e-155, -4.5159745404911819927e-172, -4.5870810097328572602e-188, -3.2905064432040069127e-204,
  9.1598554579059548847e-155, -4.5159745404911819927e-172, -4.5870810097328572602e-188, -3.2905064432040069127e-204,
  9.1598554579059548847e-155, -4.5159745404911819927e-172, -4.5870810097328572602e-188, -3.2905064432040069127e-204,
  1.7015147267057481414e-155, -4.5159745404911819927e-172, -4.5870810097328572602e-188, -3.2905064432040069127e-204,
  1.7015147267057481414e-155, -4.5159745404911819927e-172, -4.5870810097328572602e-188, -3.2905064432040069127e-204,
  1.7015147267057481414e-155, -4.5159745404911819927e-172, -4.5870810097328572602e-188, -3.2905064432040069127e-204,
  7.6922213530572229852e-156, -4.5159745404911819927e-172, -4.5870810097328572602e-188, -3.2905064432040069127e-204,
  3.0307583960570927356e-156, 5.8345524661064358191e-172, 6.9043123899963188689e-188, -3.2905064432040069127e-204,
  7.0002691755702864582e-157, 6.5928896280762691321e-173, 1.1586156901317304854e-188, -1.0100405885278530137e-205,
  7.0002691755702864582e-157, 6.5928896280762691321e-173, 1.1586156901317304854e-188, -1.0100405885278530137e-205,
  1.1734404793201255869e-157, 1.2381024895275844856e-174, -8.4789520282639751913e-191, -1.3321093418096261919e-207,
  1.1734404793201255869e-157, 1.2381024895275844856e-174, -8.4789520282639751913e-191, -1.3321093418096261919e-207,
  1.1734404793201255869e-157, 1.2381024895275844856e-174, -8.4789520282639751913e-191, -1.3321093418096261919e-207,
  4.4508689228885539715e-158, 1.2381024895275844856e-174, -8.4789520282639751913e-191, -1.3321093418096261919e-207,
  8.0910098773220302259e-159, 1.2381024895275844856e-174, -8.4789520282639751913e-191, -1.3321093418096261919e-207,
  8.0910098773220302259e-159, 1.2381024895275844856e-174, -8.4789520282639751913e-191, -1.3321093418096261919e-207,
  8.0910098773220302259e-159, 1.2381024895275844856e-174, -8.4789520282639751913e-191, -1.3321093418096261919e-207,
  3.5387999583765925506e-159, 2.2730883653953564668e-175, 2.7431118386590483722e-191, -1.3321093418096261919e-207,
  1.2626949989038732076e-159, 2.2730883653953564668e-175, 2.7431118386590483722e-191, -1.3321093418096261919e-207,
  1.2464251916751375716e-160, 6.1977249484000140293e-177, 1.1294061984896456875e-192, 2.2526486929936882202e-208,
  1.2464251916751375716e-160, 6.1977249484000140293e-177, 1.1294061984896456875e-192, 2.2526486929936882202e-208,
  1.2464251916751375716e-160, 6.1977249484000140293e-177, 1.1294061984896456875e-192, 2.2526486929936882202e-208,
  1.2464251916751375716e-160, 6.1977249484000140293e-177, 1.1294061984896456875e-192, 2.2526486929936882202e-208,
  5.3514239183991277695e-161, 6.1977249484000140293e-177, 1.1294061984896456875e-192, 2.2526486929936882202e-208,
  1.7950099192230045857e-161, -1.6991004655691153326e-177, -1.8567941091539589297e-193, -1.8074851186411640793e-209,
  1.6802919634942426156e-163, 2.8330093736631818036e-179, -7.4549709281190454638e-196, -1.4481306607622412036e-212,
  1.6802919634942426156e-163, 2.8330093736631818036e-179, -7.4549709281190454638e-196, -1.4481306607622412036e-212,
  1.6802919634942426156e-163, 2.8330093736631818036e-179, -7.4549709281190454638e-196, -1.4481306607622412036e-212,
  1.6802919634942426156e-163, 2.8330093736631818036e-179, -7.4549709281190454638e-196, -1.4481306607622412036e-212,
  1.6802919634942426156e-163, 2.8330093736631818036e-179, -7.4549709281190454638e-196, -1.4481306607622412036e-212,
  1.6802919634942426156e-163, 2.8330093736631818036e-179, -7.4549709281190454638e-196, -1.4481306607622412036e-212,
  1.6802919634942426156e-163, 2.8330093736631818036e-179, -7.4549709281190454638e-196, -1.4481306607622412036e-212,
  2.9106774506606941983e-164, 5.1948630316441287936e-180, 9.6685396110091013832e-196, 1.7562785002189357559e-211,
  2.9106774506606941983e-164, 5.1948630316441287936e-180, 9.6685396110091013832e-196, 1.7562785002189357559e-211,
  2.9106774506606941983e-164, 5.1948630316441287936e-180, 9.6685396110091013832e-196, 1.7562785002189357559e-211,
  1.1741471776254777999e-164, 1.3389912474795150614e-180, 1.106784341445028435e-196, 3.3045982549756583552e-212,
  3.0588204110786950436e-165, 3.7502330143836152136e-181, 3.6564932749519464998e-198, 3.7097125405852507464e-214,
  3.0588204110786950436e-165, 3.7502330143836152136e-181, 3.6564932749519464998e-198, 3.7097125405852507464e-214,
  8.8815756978467430465e-166, 1.3403131492807310959e-181, 3.6564932749519464998e-198, 3.7097125405852507464e-214,
  8.8815756978467430465e-166, 1.3403131492807310959e-181, 3.6564932749519464998e-198, 3.7097125405852507464e-214,
  3.4549185946116918017e-166, 1.3535321672928907047e-182, 3.1205762277848031878e-199, -3.3569248349832580936e-217,
  7.4159004299416557678e-167, 1.3535321672928907047e-182, 3.1205762277848031878e-199, -3.3569248349832580936e-217,
  7.4159004299416557678e-167, 1.3535321672928907047e-182, 3.1205762277848031878e-199, -3.3569248349832580936e-217,
  6.3257905089784152346e-168, 3.5607241064750984115e-184, -1.4832196127821708615e-201, 2.6911956484118910092e-218,
  6.3257905089784152346e-168, 3.5607241064750984115e-184, -1.4832196127821708615e-201, 2.6911956484118910092e-218,
  6.3257905089784152346e-168, 3.5607241064750984115e-184, -1.4832196127821708615e-201, 2.6911956484118910092e-218,
  6.3257905089784152346e-168, 3.5607241064750984115e-184, -1.4832196127821708615e-201, 2.6911956484118910092e-218,
  2.0862146470760309789e-168, -1.146150630053972131e-184, -1.4832196127821708615e-201, 2.6911956484118910092e-218,
  2.0862146470760309789e-168, -1.146150630053972131e-184, -1.4832196127821708615e-201, 2.6911956484118910092e-218,
  1.026320681600434562e-168, 1.2072867382105631402e-184, -1.4832196127821708615e-201, 2.6911956484118910092e-218,
  4.9637369886263658882e-169, 3.0568054078295488291e-186, 1.4980560800565460352e-202, 2.6911956484118910092e-218,
  2.3140020749373754342e-169, 3.0568054078295488291e-186, 1.4980560800565460352e-202, 2.6911956484118910092e-218,
  9.8913461809288020723e-170, 3.0568054078295488291e-186, 1.4980560800565460352e-202, 2.6911956484118910092e-218,
  3.2670088967063259373e-170, 3.0568054078295488291e-186, 1.4980560800565460352e-202, 2.6911956484118910092e-218,
  3.2670088967063259373e-170, 3.0568054078295488291e-186, 1.4980560800565460352e-202, 2.6911956484118910092e-218,
  1.6109245756507072713e-170, -6.2044048008378732802e-187, -5.4322544592823556944e-203, 4.2491789852161138683e-219,
  7.8288241512289757055e-171, 1.2181824638728806485e-186, 1.4980560800565460352e-202, 2.6911956484118910092e-218,
  3.6886133485899290404e-171, 2.9887099189454666024e-187, 4.774153170641553462e-203, 4.2491789852161138683e-219,
  1.6185079472704052482e-171, 2.9887099189454666024e-187, 4.774153170641553462e-203, 4.2491789852161138683e-219,
  5.8345524661064358191e-172, 6.9043123899963188689e-188, -3.2905064432040069127e-204, -9.1795828160190082842e-224,
  6.5928896280762691321e-173, 1.1586156901317304854e-188, -1.0100405885278530137e-205, -9.1795828160190082842e-224,
  6.5928896280762691321e-173, 1.1586156901317304854e-188, -1.0100405885278530137e-205, -9.1795828160190082842e-224,
  6.5928896280762691321e-173, 1.1586156901317304854e-188, -1.0100405885278530137e-205, -9.1795828160190082842e-224,
  1.2381024895275844856e-174, -8.4789520282639751913e-191, -1.332109341809626019e-207, -9.1795828160190082842e-224,
  1.2381024895275844856e-174, -8.4789520282639751913e-191, -1.332109341809626019e-207, -9.1795828160190082842e-224,
  1.2381024895275844856e-174, -8.4789520282639751913e-191, -1.332109341809626019e-207, -9.1795828160190082842e-224,
  1.2381024895275844856e-174, -8.4789520282639751913e-191, -1.332109341809626019e-207, -9.1795828160190082842e-224,
  1.2381024895275844856e-174, -8.4789520282639751913e-191, -1.332109341809626019e-207, -9.1795828160190082842e-224,
  1.2381024895275844856e-174, -8.4789520282639751913e-191, -1.332109341809626019e-207, -9.1795828160190082842e-224,
  2.2730883653953564668e-175, 2.7431118386590483722e-191, -1.332109341809626019e-207, -9.1795828160190082842e-224,
  2.2730883653953564668e-175, 2.7431118386590483722e-191, -1.332109341809626019e-207, -9.1795828160190082842e-224,
  2.2730883653953564668e-175, 2.7431118386590483722e-191, -1.332109341809626019e-207, -9.1795828160190082842e-224,
  1.0095962991602958391e-175, -6.2404128071707654958e-193, 3.0593092910744445285e-209, 5.4622616159087170031e-225,
  3.7785026604276538491e-176, -6.2404128071707654958e-193, 3.0593092910744445285e-209, 5.4622616159087170031e-225,
  6.1977249484000140293e-177, 1.1294061984896456875e-192, 2.2526486929936882202e-208, -5.3441928036578162463e-225,
  6.1977249484000140293e-177, 1.1294061984896456875e-192, 2.2526486929936882202e-208, -5.3441928036578162463e-225,
  6.1977249484000140293e-177, 1.1294061984896456875e-192, 2.2526486929936882202e-208, -5.3441928036578162463e-225,
  2.2493122414154495675e-177, 2.5268245888628466632e-193, 3.0593092910744445285e-209, 5.4622616159087170031e-225,
  2.7510588792316711745e-178, 3.3501523985444386676e-194, 6.2591208621664049475e-210, 5.9034406125450500218e-227,
  2.7510588792316711745e-178, 3.3501523985444386676e-194, 6.2591208621664049475e-210, 5.9034406125450500218e-227,
  2.7510588792316711745e-178, 3.3501523985444386676e-194, 6.2591208621664049475e-210, 5.9034406125450500218e-227,
  2.8330093736631818036e-179, -7.4549709281190454638e-196, -1.4481306607622412036e-212, 9.9192633285681635836e-229,
  2.8330093736631818036e-179, -7.4549709281190454638e-196, -1.4481306607622412036e-212, 9.9192633285681635836e-229,
  2.8330093736631818036e-179, -7.4549709281190454638e-196, -1.4481306607622412036e-212, 9.9192633285681635836e-229,
  2.8330093736631818036e-179, -7.4549709281190454638e-196, -1.4481306607622412036e-212, 9.9192633285681635836e-229,
  1.2906606599973359683e-179, -7.4549709281190454638e-196, -1.4481306607622412036e-212, 9.9192633285681635836e-229,
  5.1948630316441287936e-180, 9.6685396110091013832e-196, 1.7562785002189355449e-211, 1.6821693549018732055e-227,
  1.3389912474795150614e-180, 1.106784341445028435e-196, 3.3045982549756578275e-212, 6.2685154049107876715e-228,
  1.3389912474795150614e-180, 1.106784341445028435e-196, 3.3045982549756578275e-212, 6.2685154049107876715e-228,
  3.7502330143836152136e-181, 3.6564932749519464998e-198, 3.7097125405852507464e-214, 2.5658818466966882188e-231,
  3.7502330143836152136e-181, 3.6564932749519464998e-198, 3.7097125405852507464e-214, 2.5658818466966882188e-231,
  1.3403131492807310959e-181, 3.6564932749519464998e-198, 3.7097125405852507464e-214, 2.5658818466966882188e-231,
  1.3535321672928907047e-182, 3.1205762277848031878e-199, -3.3569248349832580936e-217, -1.0577661142165146927e-233,
  1.3535321672928907047e-182, 3.1205762277848031878e-199, -3.3569248349832580936e-217, -1.0577661142165146927e-233,
  1.3535321672928907047e-182, 3.1205762277848031878e-199, -3.3569248349832580936e-217, -1.0577661142165146927e-233,
  1.3535321672928907047e-182, 3.1205762277848031878e-199, -3.3569248349832580936e-217, -1.0577661142165146927e-233,
  6.0043220944823941786e-183, 3.1205762277848031878e-199, -3.3569248349832580936e-217, -1.0577661142165146927e-233,
  2.2388223052591377446e-183, 3.1205762277848031878e-199, -3.3569248349832580936e-217, -1.0577661142165146927e-233,
  3.5607241064750984115e-184, -1.4832196127821708615e-201, 2.6911956484118910092e-218, -5.1336618966962585332e-235,
  3.5607241064750984115e-184, -1.4832196127821708615e-201, 2.6911956484118910092e-218, -5.1336618966962585332e-235,
  3.5607241064750984115e-184, -1.4832196127821708615e-201, 2.6911956484118910092e-218, -5.1336618966962585332e-235,
  1.2072867382105631402e-184, -1.4832196127821708615e-201, 2.6911956484118910092e-218, -5.1336618966962585332e-235,
  3.0568054078295488291e-186, 1.4980560800565460352e-202, 2.6911956484118910092e-218, -5.1336618966962585332e-235,
  3.0568054078295488291e-186, 1.4980560800565460352e-202, 2.6911956484118910092e-218, -5.1336618966962585332e-235,
  3.0568054078295488291e-186, 1.4980560800565460352e-202, 2.6911956484118910092e-218, -5.1336618966962585332e-235,
  3.0568054078295488291e-186, 1.4980560800565460352e-202, 2.6911956484118910092e-218, -5.1336618966962585332e-235,
  3.0568054078295488291e-186, 1.4980560800565460352e-202, 2.6911956484118910092e-218, -5.1336618966962585332e-235,
  3.0568054078295488291e-186, 1.4980560800565460352e-202, 2.6911956484118910092e-218, -5.1336618966962585332e-235,
  1.2181824638728806485e-186, 1.4980560800565460352e-202, 2.6911956484118910092e-218, -5.1336618966962585332e-235,
  2.9887099189454666024e-187, 4.774153170641553462e-203, 4.2491789852161132393e-219, 7.4467067939231424594e-235,
  2.9887099189454666024e-187, 4.774153170641553462e-203, 4.2491789852161132393e-219, 7.4467067939231424594e-235,
  6.9043123899963188689e-188, -3.2905064432040069127e-204, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  6.9043123899963188689e-188, -3.2905064432040069127e-204, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  1.1586156901317304854e-188, -1.0100405885278530137e-205, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  1.1586156901317304854e-188, -1.0100405885278530137e-205, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  1.1586156901317304854e-188, -1.0100405885278530137e-205, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  4.4040360264865697732e-189, -1.0100405885278530137e-205, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  8.129755890712020335e-190, 9.8339840169166049336e-206, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  8.129755890712020335e-190, 9.8339840169166049336e-206, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  8.129755890712020335e-190, 9.8339840169166049336e-206, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  3.6409303439428119063e-190, -1.332109341809626019e-207, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  1.3965175705582071936e-190, -1.332109341809626019e-207, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  2.7431118386590483722e-191, -1.332109341809626019e-207, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  2.7431118386590483722e-191, -1.332109341809626019e-207, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  2.7431118386590483722e-191, -1.332109341809626019e-207, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  1.3403538552936701153e-191, 1.7826390804083638359e-207, -9.1795828160190063645e-224, -2.3569545504732004486e-239,
  6.389748636109812983e-192, 2.2526486929936882202e-208, -5.3441928036578156465e-225, -7.741539335184153052e-241,
  2.8828536776963681193e-192, 2.2526486929936882202e-208, -5.3441928036578156465e-225, -7.741539335184153052e-241,
  1.1294061984896456875e-192, 2.2526486929936882202e-208, -5.3441928036578156465e-225, -7.741539335184153052e-241,
  2.5268245888628466632e-193, 3.0593092910744445285e-209, 5.4622616159087170031e-225, 4.2560351759808952526e-241,
  2.5268245888628466632e-193, 3.0593092910744445285e-209, 5.4622616159087170031e-225, 4.2560351759808952526e-241,
  3.3501523985444386676e-194, 6.2591208621664049475e-210, 5.9034406125450490845e-227, 1.3186893776791012681e-242,
  3.3501523985444386676e-194, 6.2591208621664049475e-210, 5.9034406125450490845e-227, 1.3186893776791012681e-242,
  3.3501523985444386676e-194, 6.2591208621664049475e-210, 5.9034406125450490845e-227, 1.3186893776791012681e-242,
  6.1039071228393547627e-195, 1.7562785002189355449e-211, 1.6821693549018732055e-227, -8.7276385348052817035e-244,
  6.1039071228393547627e-195, 1.7562785002189355449e-211, 1.6821693549018732055e-227, -8.7276385348052817035e-244,
  6.1039071228393547627e-195, 1.7562785002189355449e-211, 1.6821693549018732055e-227, -8.7276385348052817035e-244,
  2.6792050150137250131e-195, 1.7562785002189355449e-211, 1.6821693549018732055e-227, -8.7276385348052817035e-244,
  9.6685396110091013832e-196, 1.7562785002189355449e-211, 1.6821693549018732055e-227, -8.7276385348052817035e-244,
  2.0416567491425607157e-177, 6.0959078275963141821e-193, 1.156336993964950812e-208, 2.7126166236326293347e-224,
  2.0416567491425607157e-177, 6.0959078275963141821e-193, 1.156336993964950812e-208, 2.7126166236326293347e-224,
  2.0416567491425607157e-177, 6.0959078275963141821e-193, 1.156336993964950812e-208, 2.7126166236326293347e-224,
  6.7450395650278649168e-179, 6.8432117823206978686e-195, 4.7332165749391048364e-212, 4.4984059688774601837e-228,
  6.7450395650278649168e-179, 6.8432117823206978686e-195, 4.7332165749391048364e-212, 4.4984059688774601837e-228,
  6.7450395650278649168e-179, 6.8432117823206978686e-195, 4.7332165749391048364e-212, 4.4984059688774601837e-228,
  6.7450395650278649168e-179, 6.8432117823206978686e-195, 4.7332165749391048364e-212, 4.4984059688774601837e-228,
  6.7450395650278649168e-179, 6.8432117823206978686e-195, 4.7332165749391048364e-212, 4.4984059688774601837e-228,
  5.756447103644822603e-180, -6.1924333305615830735e-198, -1.9512340798794268979e-214, -3.6162764918921697356e-230,
  5.756447103644822603e-180, -6.1924333305615830735e-198, -1.9512340798794268979e-214, -3.6162764918921697356e-230,
  5.756447103644822603e-180, -6.1924333305615830735e-198, -1.9512340798794268979e-214, -3.6162764918921697356e-230,
  5.756447103644822603e-180, -6.1924333305615830735e-198, -1.9512340798794268979e-214, -3.6162764918921697356e-230,
  1.9005753194802080146e-180, -6.1924333305615830735e-198, -1.9512340798794268979e-214, -3.6162764918921697356e-230,
  1.9005753194802080146e-180, -6.1924333305615830735e-198, -1.9512340798794268979e-214, -3.6162764918921697356e-230,
  9.3660737343905436753e-181, -6.1924333305615830735e-198, -1.9512340798794268979e-214, -3.6162764918921697356e-230,
  4.5462340041847754398e-181, -6.1924333305615830735e-198, -1.9512340798794268979e-214, -3.6162764918921697356e-230,
  2.1363141390818913221e-181, -6.1924333305615830735e-198, -1.9512340798794268979e-214, -3.6162764918921697356e-230,
  9.3135420653044926323e-182, -6.1924333305615830735e-198, -1.9512340798794268979e-214, -3.6162764918921697356e-230,
  3.2887424025472810002e-182, 7.185309278132283136e-198, -1.9512340798794268979e-214, -3.6162764918921697356e-230,
  2.7634257116867652192e-183, 4.9643797378534984559e-199, -9.4699347169310243473e-216, -9.2331809177749095611e-233,
  2.7634257116867652192e-183, 4.9643797378534984559e-199, -9.4699347169310243473e-216, -9.2331809177749095611e-233,
  2.7634257116867652192e-183, 4.9643797378534984559e-199, -9.4699347169310243473e-216, -9.2331809177749095611e-233,
  2.7634257116867652192e-183, 4.9643797378534984559e-199, -9.4699347169310243473e-216, -9.2331809177749095611e-233,
  8.806758170751374203e-184, 7.8383517263666503337e-200, 1.3736749441945438342e-215, -9.2331809177749095611e-233,
  8.806758170751374203e-184, 7.8383517263666503337e-200, 1.3736749441945438342e-215, -9.2331809177749095611e-233,
  4.0998834342223036605e-184, 7.8383517263666503337e-200, 1.3736749441945438342e-215, -9.2331809177749095611e-233,
  1.7464460659577689118e-184, 2.612671019845610006e-200, 2.1334073625072069974e-216, -9.2331809177749095611e-233,
  5.697273818255015375e-185, -1.6933341491052464293e-204, -4.3478137385944270631e-220, -2.3353910329236990725e-236,
  5.697273818255015375e-185, -1.6933341491052464293e-204, -4.3478137385944270631e-220, -2.3353910329236990725e-236,
  2.755477107924346286e-185, -1.6933341491052464293e-204, -4.3478137385944270631e-220, -2.3353910329236990725e-236,
  1.2845787527590117414e-185, -1.6933341491052464293e-204, -4.3478137385944270631e-220, -2.3353910329236990725e-236,
  5.4912957517634446918e-186, -1.6933341491052464293e-204, -4.3478137385944270631e-220, -2.3353910329236990725e-236,
  1.8140498638501083305e-186, -1.6933341491052464293e-204, -4.3478137385944270631e-220, -2.3353910329236990725e-236,
  1.8140498638501083305e-186, -1.6933341491052464293e-204, -4.3478137385944270631e-220, -2.3353910329236990725e-236,
  8.9473839187177424013e-187, -1.6933341491052464293e-204, -4.3478137385944270631e-220, -2.3353910329236990725e-236,
  4.3508265588260719497e-187, -1.6933341491052464293e-204, -4.3478137385944270631e-220, -2.3353910329236990725e-236,
  2.0525478788802367239e-187, -1.6933341491052464293e-204, -4.3478137385944270631e-220, -2.3353910329236990725e-236,
  9.0340853890731911095e-188, -1.6933341491052464293e-204, -4.3478137385944270631e-220, -2.3353910329236990725e-236,
  3.288388689208603045e-188, -1.6933341491052464293e-204, -4.3478137385944270631e-220, -2.3353910329236990725e-236,
  4.1554033927630885323e-189, -9.8582956929636044137e-206, -1.4280619485269765742e-221, 1.2171222696290252021e-237,
  4.1554033927630885323e-189, -9.8582956929636044137e-206, -1.4280619485269765742e-221, 1.2171222696290252021e-237,
  4.1554033927630885323e-189, -9.8582956929636044137e-206, -1.4280619485269765742e-221, 1.2171222696290252021e-237,
  5.643429553477207926e-190, 1.0076094209231528444e-205, 7.8509991660024955813e-222, 1.2171222696290252021e-237,
  5.643429553477207926e-190, 1.0076094209231528444e-205, 7.8509991660024955813e-222, 1.2171222696290252021e-237,
  5.643429553477207926e-190, 1.0076094209231528444e-205, 7.8509991660024955813e-222, 1.2171222696290252021e-237,
  1.1546040067079994973e-190, 1.0889925813396166947e-207, 2.4325525462765697993e-223, -1.1429360314275701698e-239,
  1.1546040067079994973e-190, 1.0889925813396166947e-207, 2.4325525462765697993e-223, -1.1429360314275701698e-239,
  3.2397620015697148712e-192, 3.1030547578511949035e-208, -1.609965144193984205e-224, -1.8313007053436627876e-240,
  3.2397620015697148712e-192, 3.1030547578511949035e-208, -1.609965144193984205e-224, -1.8313007053436627876e-240,
  3.2397620015697148712e-192, 3.1030547578511949035e-208, -1.609965144193984205e-224, -1.8313007053436627876e-240,
  3.2397620015697148712e-192, 3.1030547578511949035e-208, -1.609965144193984205e-224, -1.8313007053436627876e-240,
  3.2397620015697148712e-192, 3.1030547578511949035e-208, -1.609965144193984205e-224, -1.8313007053436627876e-240,
  3.2397620015697148712e-192, 3.1030547578511949035e-208, -1.609965144193984205e-224, -1.8313007053436627876e-240,
  1.4863145223629928288e-192, -7.9038076992129241506e-209, -1.609965144193984205e-224, -1.8313007053436627876e-240,
  6.0959078275963141821e-193, 1.156336993964950812e-208, 2.7126166236326293347e-224, -1.8313007053436627876e-240,
  1.712289129579509076e-193, 1.8297811202182925249e-209, 1.1003018740995688645e-226, 5.827891678485165325e-243,
  1.712289129579509076e-193, 1.8297811202182925249e-209, 1.1003018740995688645e-226, 5.827891678485165325e-243,
  6.1638445507530779946e-194, -6.0361608463951204924e-210, 1.1003018740995688645e-226, 5.827891678485165325e-243,
  6.8432117823206978686e-195, 4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.029900079464340522e-245,
  6.8432117823206978686e-195, 4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.029900079464340522e-245,
  6.8432117823206978686e-195, 4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.029900079464340522e-245,
  6.8432117823206978686e-195, 4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.029900079464340522e-245,
  3.418509674495068119e-195, 4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.029900079464340522e-245,
  1.7061586205822532442e-195, 4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.029900079464340522e-245,
  8.499830936258458068e-196, 4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.029900079464340522e-245,
  4.218953301476420881e-196, 4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.029900079464340522e-245,
  2.0785144840854027628e-196, -1.9512340798794268979e-214, -3.6162764918921692779e-230, -2.8387319855193022476e-246,
  1.008295075389893466e-196, -1.9512340798794268979e-214, -3.6162764918921692779e-230, -2.8387319855193022476e-246,
  4.7318537104213881764e-197, -1.9512340798794268979e-214, -3.6162764918921692779e-230, -2.8387319855193022476e-246,
  2.0563051886826149345e-197, -1.9512340798794268979e-214, -3.6162764918921692779e-230, -2.8387319855193022476e-246,
  7.185309278132283136e-198, -1.9512340798794268979e-214, -3.6162764918921692779e-230, -2.8387319855193022476e-246,
  4.9643797378534984559e-199, -9.4699347169310243473e-216, -9.2331809177749077733e-233, -1.4042876247421728101e-248,
  4.9643797378534984559e-199, -9.4699347169310243473e-216, -9.2331809177749077733e-233, -1.4042876247421728101e-248,
  4.9643797378534984559e-199, -9.4699347169310243473e-216, -9.2331809177749077733e-233, -1.4042876247421728101e-248,
  4.9643797378534984559e-199, -9.4699347169310243473e-216, -9.2331809177749077733e-233, -1.4042876247421728101e-248,
  7.8383517263666503337e-200, 1.3736749441945438342e-215, -9.2331809177749077733e-233, -1.4042876247421728101e-248,
  7.8383517263666503337e-200, 1.3736749441945438342e-215, -9.2331809177749077733e-233, -1.4042876247421728101e-248,
  7.8383517263666503337e-200, 1.3736749441945438342e-215, -9.2331809177749077733e-233, -1.4042876247421728101e-248,
  2.612671019845610006e-200, 2.1334073625072069974e-216, -9.2331809177749077733e-233, -1.4042876247421728101e-248,
  2.612671019845610006e-200, 2.1334073625072069974e-216, -9.2331809177749077733e-233, -1.4042876247421728101e-248,
  1.306250843215349634e-200, 2.1334073625072069974e-216, -9.2331809177749077733e-233, -1.4042876247421728101e-248,
  6.5304075490021959302e-201, 6.8298960257742791824e-217, 6.8696910062179237095e-233, 3.8349029251851101018e-249,
  3.2643571074265457254e-201, -4.2219277387461470355e-218, -1.753154605289404553e-234, -7.5861268822635538093e-251,
  1.6313318866387202604e-201, -4.2219277387461470355e-218, -1.753154605289404553e-234, -7.5861268822635538093e-251,
  8.1481927624480752786e-202, -4.2219277387461470355e-218, -1.753154605289404553e-234, -7.5861268822635538093e-251,
  4.0656297104785107096e-202, 4.8431832608149701961e-218, 8.3111403472061145651e-234, 1.6001805286092554504e-249,
  2.0243481844937293316e-202, 3.1062776103441183191e-219, 7.6291913283447536617e-235, 2.0347903074934629333e-250,
  1.0037074215013384159e-202, 3.1062776103441183191e-219, 7.6291913283447536617e-235, 2.0347903074934629333e-250,
  4.9338704000514295811e-203, 3.1062776103441183191e-219, 7.6291913283447536617e-235, 2.0347903074934629333e-250,
  2.3822684925704522921e-203, 3.1062776103441183191e-219, 7.6291913283447536617e-235, 2.0347903074934629333e-250,
  1.1064675388299639308e-203, 2.7343042298126957741e-220, 5.5273393987134252385e-236, 1.1432574793608782288e-251,
  4.6856706195971960852e-204, 2.7343042298126957741e-220, 5.5273393987134252385e-236, 1.1432574793608782288e-251,
  1.4961682352459748279e-204, -8.0675475439086544798e-221, -3.6970842501441777651e-237, -5.7032870362481275794e-253,
  1.4961682352459748279e-204, -8.0675475439086544798e-221, -3.6970842501441777651e-237, -5.7032870362481275794e-253,
  6.9879263915816924805e-205, 9.6377473771091526132e-221, 1.5959741828948633012e-236, 2.7031904319843495713e-252,
  3.0010484111426663515e-205, 7.8509991660024955813e-222, 1.2171222696290252021e-237, -2.4742181023285720738e-254,
  1.0076094209231528444e-205, 7.8509991660024955813e-222, 1.2171222696290252021e-237, -2.4742181023285720738e-254,
  1.0889925813396166947e-207, 2.4325525462765697993e-223, -1.1429360314275701698e-239, 8.3218722366085688343e-256,
  1.0889925813396166947e-207, 2.4325525462765697993e-223, -1.1429360314275701698e-239, 8.3218722366085688343e-256,
  1.0889925813396166947e-207, 2.4325525462765697993e-223, -1.1429360314275701698e-239, 8.3218722366085688343e-256,
  1.0889925813396166947e-207, 2.4325525462765697993e-223, -1.1429360314275701698e-239, 8.3218722366085688343e-256,
  1.0889925813396166947e-207, 2.4325525462765697993e-223, -1.1429360314275701698e-239, 8.3218722366085688343e-256,
  1.0889925813396166947e-207, 2.4325525462765697993e-223, -1.1429360314275701698e-239, 8.3218722366085688343e-256,
  1.0889925813396166947e-207, 2.4325525462765697993e-223, -1.1429360314275701698e-239, 8.3218722366085688343e-256,
  3.1030547578511949035e-208, -1.609965144193984205e-224, -1.8313007053436625212e-240, -2.3341145329525059632e-256,
  3.1030547578511949035e-208, -1.609965144193984205e-224, -1.8313007053436625212e-240, -2.3341145329525059632e-256,
  1.156336993964950812e-208, 2.7126166236326293347e-224, -1.8313007053436625212e-240, -2.3341145329525059632e-256,
  1.8297811202182925249e-209, 1.1003018740995688645e-226, 5.827891678485165325e-243, -3.1174271110208206547e-259,
  1.8297811202182925249e-209, 1.1003018740995688645e-226, 5.827891678485165325e-243, -3.1174271110208206547e-259,
  1.8297811202182925249e-209, 1.1003018740995688645e-226, 5.827891678485165325e-243, -3.1174271110208206547e-259,
  6.1308251778939023781e-210, 1.1003018740995688645e-226, 5.827891678485165325e-243, -3.1174271110208206547e-259,
  4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.0299000794643401155e-245, -2.8075477999879273582e-261,
  4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.0299000794643401155e-245, -2.8075477999879273582e-261,
  4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.0299000794643401155e-245, -2.8075477999879273582e-261,
  4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.0299000794643401155e-245, -2.8075477999879273582e-261,
  4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.0299000794643401155e-245, -2.8075477999879273582e-261,
  4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.0299000794643401155e-245, -2.8075477999879273582e-261,
  4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.0299000794643401155e-245, -2.8075477999879273582e-261,
  4.7332165749391048364e-212, 4.4984059688774601837e-228, -3.0299000794643401155e-245, -2.8075477999879273582e-261,
  2.3568521170701555846e-212, -7.7818310317651142243e-229, -3.0299000794643401155e-245, -2.8075477999879273582e-261,
  1.1686698881356804311e-212, 1.8601114328504743806e-228, -3.0299000794643401155e-245, -2.8075477999879273582e-261,
  5.7457877366844311816e-213, 5.409641648369814791e-229, -3.0299000794643401155e-245, -2.8075477999879273582e-261,
  2.7753321643482446169e-213, -1.1860946916976500828e-229, 6.3146909508553973881e-246, 1.2573885592501532045e-261,
  1.290104378180150675e-213, 2.1117734783360818049e-229, 4.2928382696354204061e-245, -2.8075477999879273582e-261,
  5.4749048509610403382e-214, 4.6283939331921604413e-230, 6.3146909508553973881e-246, 1.2573885592501532045e-261,
  1.7618353855408067201e-214, 5.060587206499956961e-231, 5.9380161562121075096e-247, -1.2904053011746964278e-263,
  1.7618353855408067201e-214, 5.060587206499956961e-231, 5.9380161562121075096e-247, -1.2904053011746964278e-263,
  8.3356801918574821257e-215, 5.060587206499956961e-231, 5.9380161562121075096e-247, -1.2904053011746964278e-263,
  3.6943433600821895879e-215, 5.060587206499956961e-231, 5.9380161562121075096e-247, -1.2904053011746964278e-263,
  1.3736749441945438342e-215, -9.2331809177749077733e-233, -1.4042876247421726117e-248, -9.9505977179164858712e-265,
  2.1334073625072069974e-216, -9.2331809177749077733e-233, -1.4042876247421726117e-248, -9.9505977179164858712e-265,
  2.1334073625072069974e-216, -9.2331809177749077733e-233, -1.4042876247421726117e-248, -9.9505977179164858712e-265,
  2.1334073625072069974e-216, -9.2331809177749077733e-233, -1.4042876247421726117e-248, -9.9505977179164858712e-265,
  6.8298960257742791824e-217, 6.8696910062179237095e-233, 3.8349029251851101018e-249, -2.6436684620390282645e-267,
  6.8298960257742791824e-217, 6.8696910062179237095e-233, 3.8349029251851101018e-249, -2.6436684620390282645e-267,
  3.2038516259498326923e-217, -1.1817449557784924788e-233, -6.3454186796659920093e-250, -2.6436684620390282645e-267,
  1.3908294260376086421e-217, 2.8439730252197153919e-233, 3.8349029251851101018e-249, -2.6436684620390282645e-267,
  4.8431832608149701961e-218, 8.3111403472061145651e-234, 1.6001805286092554504e-249, -2.6436684620390282645e-267,
  3.1062776103441183191e-219, 7.6291913283447536617e-235, 2.0347903074934629333e-250, -2.6436684620390282645e-267,
  3.1062776103441183191e-219, 7.6291913283447536617e-235, 2.0347903074934629333e-250, -2.6436684620390282645e-267,
  3.1062776103441183191e-219, 7.6291913283447536617e-235, 2.0347903074934629333e-250, -2.6436684620390282645e-267,
  3.1062776103441183191e-219, 7.6291913283447536617e-235, 2.0347903074934629333e-250, -2.6436684620390282645e-267,
  2.7343042298126957741e-220, 5.5273393987134252385e-236, 1.1432574793608780349e-251, 1.2329569415922591084e-267,
  2.7343042298126957741e-220, 5.5273393987134252385e-236, 1.1432574793608780349e-251, 1.2329569415922591084e-267,
  2.7343042298126957741e-220, 5.5273393987134252385e-236, 1.1432574793608780349e-251, 1.2329569415922591084e-267,
  2.7343042298126957741e-220, 5.5273393987134252385e-236, 1.1432574793608780349e-251, 1.2329569415922591084e-267,
  9.6377473771091526132e-221, 1.5959741828948633012e-236, 2.7031904319843490867e-252, 2.638005906844372114e-268,
  7.8509991660024955813e-222, 1.2171222696290252021e-237, -2.4742181023285720738e-254, -1.2030990169203137715e-270,
  7.8509991660024955813e-222, 1.2171222696290252021e-237, -2.4742181023285720738e-254, -1.2030990169203137715e-270,
  7.8509991660024955813e-222, 1.2171222696290252021e-237, -2.4742181023285720738e-254, -1.2030990169203137715e-270,
  7.8509991660024955813e-222, 1.2171222696290252021e-237, -2.4742181023285720738e-254, -1.2030990169203137715e-270,
  2.318094503184431479e-222, -1.1429360314275701698e-239, 8.3218722366085688343e-256, -2.0046830753539155726e-272,
  2.318094503184431479e-222, -1.1429360314275701698e-239, 8.3218722366085688343e-256, -2.0046830753539155726e-272,
  9.3486833747991514629e-223, -1.1429360314275701698e-239, 8.3218722366085688343e-256, -2.0046830753539155726e-272,
  2.4325525462765697993e-223, -1.1429360314275701698e-239, 8.3218722366085688343e-256, -2.0046830753539155726e-272,
  2.4325525462765697993e-223, -1.1429360314275701698e-239, 8.3218722366085688343e-256, -2.0046830753539155726e-272,
  7.0351983914592419146e-224, 7.766758903588374524e-240, 8.3218722366085688343e-256, -2.0046830753539155726e-272,
  7.0351983914592419146e-224, 7.766758903588374524e-240, 8.3218722366085688343e-256, -2.0046830753539155726e-272,
  2.7126166236326293347e-224, -1.8313007053436625212e-240, -2.3341145329525056675e-256, -2.0046830753539155726e-272,
  5.5132573971932232487e-225, 5.6821419688934674008e-241, 3.2988215943776273615e-257, 2.1353977370878701046e-273,
  5.5132573971932232487e-225, 5.6821419688934674008e-241, 3.2988215943776273615e-257, 2.1353977370878701046e-273,
  1.1003018740995688645e-226, 5.827891678485165325e-243, -3.117427111020820077e-259, -5.9718623963762788119e-275,
  1.1003018740995688645e-226, 5.827891678485165325e-243, -3.117427111020820077e-259, -5.9718623963762788119e-275,
  1.1003018740995688645e-226, 5.827891678485165325e-243, -3.117427111020820077e-259, -5.9718623963762788119e-275,
  1.1003018740995688645e-226, 5.827891678485165325e-243, -3.117427111020820077e-259, -5.9718623963762788119e-275,
  1.1003018740995688645e-226, 5.827891678485165325e-243, -3.117427111020820077e-259, -5.9718623963762788119e-275,
  1.1003018740995688645e-226, 5.827891678485165325e-243, -3.117427111020820077e-259, -5.9718623963762788119e-275,
  2.560476225709334075e-227, 5.827891678485165325e-243, -3.117427111020820077e-259, -5.9718623963762788119e-275,
  2.560476225709334075e-227, 5.827891678485165325e-243, -3.117427111020820077e-259, -5.9718623963762788119e-275,
  4.4984059688774601837e-228, -3.0299000794643401155e-245, -2.8075477999879273582e-261, -1.472095602234059958e-277,
  4.4984059688774601837e-228, -3.0299000794643401155e-245, -2.8075477999879273582e-261, -1.472095602234059958e-277,
  4.4984059688774601837e-228, -3.0299000794643401155e-245, -2.8075477999879273582e-261, -1.472095602234059958e-277,
  1.8601114328504743806e-228, -3.0299000794643401155e-245, -2.8075477999879273582e-261, -1.472095602234059958e-277,
  5.409641648369814791e-229, -3.0299000794643401155e-245, -2.8075477999879273582e-261, -1.472095602234059958e-277,
  5.409641648369814791e-229, -3.0299000794643401155e-245, -2.8075477999879273582e-261, -1.472095602234059958e-277,
  2.1117734783360818049e-229, 4.2928382696354204061e-245, -2.8075477999879273582e-261, -1.472095602234059958e-277,
  4.6283939331921604413e-230, 6.3146909508553973881e-246, 1.2573885592501529789e-261, 3.0408903374280139822e-277,
  4.6283939331921604413e-230, 6.3146909508553973881e-246, 1.2573885592501529789e-261, 3.0408903374280139822e-277,
  5.060587206499956961e-231, 5.9380161562121075096e-247, -1.2904053011746964278e-263, 8.7279092175580820317e-280,
  5.060587206499956961e-231, 5.9380161562121075096e-247, -1.2904053011746964278e-263, 8.7279092175580820317e-280,
  5.060587206499956961e-231, 5.9380161562121075096e-247, -1.2904053011746964278e-263, 8.7279092175580820317e-280,
  5.060587206499956961e-231, 5.9380161562121075096e-247, -1.2904053011746964278e-263, 8.7279092175580820317e-280,
  2.4841276986611042098e-231, 2.1712682097791944335e-248, 2.9746046415267896827e-264, -8.6516445844406224413e-282,
  1.1958979447416775482e-231, 2.1712682097791944335e-248, 2.9746046415267896827e-264, -8.6516445844406224413e-282,
  5.5178306778196421733e-232, 2.1712682097791944335e-248, 2.9746046415267896827e-264, -8.6516445844406224413e-282,
  2.2972562930210755192e-232, 2.1712682097791944335e-248, 2.9746046415267896827e-264, -8.6516445844406224413e-282,
  6.8696910062179237095e-233, 3.8349029251851101018e-249, -2.6436684620390282645e-267, -4.3807022524130141006e-284,
  6.8696910062179237095e-233, 3.8349029251851101018e-249, -2.6436684620390282645e-267, -4.3807022524130141006e-284,
  2.8439730252197153919e-233, 3.8349029251851101018e-249, -2.6436684620390282645e-267, -4.3807022524130141006e-284,
  8.3111403472061145651e-234, 1.6001805286092554504e-249, -2.6436684620390282645e-267, -4.3807022524130141006e-284,
  8.3111403472061145651e-234, 1.6001805286092554504e-249, -2.6436684620390282645e-267, -4.3807022524130141006e-284,
  3.2789928709583552854e-234, 4.8281933032132812475e-250, -2.6436684620390282645e-267, -4.3807022524130141006e-284,
  7.6291913283447536617e-235, 2.0347903074934629333e-250, -2.6436684620390282645e-267, -4.3807022524130141006e-284,
  7.6291913283447536617e-235, 2.0347903074934629333e-250, -2.6436684620390282645e-267, -4.3807022524130141006e-284,
  1.3390069830350552605e-235, -6.026193929640082176e-252, -7.0535576022338457803e-268, -4.3807022524130141006e-284,
  1.3390069830350552605e-235, -6.026193929640082176e-252, -7.0535576022338457803e-268, -4.3807022524130141006e-284,
  1.3390069830350552605e-235, -6.026193929640082176e-252, -7.0535576022338457803e-268, -4.3807022524130141006e-284,
  5.5273393987134252385e-236, 1.1432574793608780349e-251, 1.2329569415922591084e-267, -4.3807022524130141006e-284,
  1.5959741828948633012e-236, 2.7031904319843490867e-252, 2.638005906844371576e-268, 6.3790946999826013345e-284,
  1.5959741828948633012e-236, 2.7031904319843490867e-252, 2.638005906844371576e-268, 6.3790946999826013345e-284,
  6.1313287894022281692e-237, 5.2084434157824127104e-253, 2.1511502957481757317e-269, 3.2670891426006739096e-285,
  1.2171222696290252021e-237, -2.4742181023285720738e-254, -1.2030990169203137715e-270, -9.5347405022956042207e-287,
  1.2171222696290252021e-237, -2.4742181023285720738e-254, -1.2030990169203137715e-270, -9.5347405022956042207e-287,
  1.2171222696290252021e-237, -2.4742181023285720738e-254, -1.2030990169203137715e-270, -9.5347405022956042207e-287,
  6.0284645465737476297e-238, -2.4742181023285720738e-254, -1.2030990169203137715e-270, -9.5347405022956042207e-287,
  2.9570854717154947523e-238, 4.3456134301905148502e-254, 6.3684349745470443788e-270, -9.5347405022956042207e-287,
  1.4213959342863689955e-238, 9.3569766393097138822e-255, 2.5826679788133653036e-270, -9.5347405022956042207e-287,
  6.5355116557180594664e-239, 9.3569766393097138822e-255, 2.5826679788133653036e-270, -9.5347405022956042207e-287,
  2.6962878121452450746e-239, 8.3218722366085688343e-256, -2.0046830753539152442e-272, -3.4057806738724185961e-288,
  7.766758903588374524e-240, 8.3218722366085688343e-256, -2.0046830753539152442e-272, -3.4057806738724185961e-288,
  7.766758903588374524e-240, 8.3218722366085688343e-256, -2.0046830753539152442e-272, -3.4057806738724185961e-288,
  2.9677290991223565342e-240, -2.3341145329525056675e-256, -2.0046830753539152442e-272, -3.4057806738724185961e-288,
  5.6821419688934674008e-241, 3.2988215943776273615e-257, 2.1353977370878701046e-273, -1.2215123283371736879e-289,
  5.6821419688934674008e-241, 3.2988215943776273615e-257, 2.1353977370878701046e-273, -1.2215123283371736879e-289,
  5.6821419688934674008e-241, 3.2988215943776273615e-257, 2.1353977370878701046e-273, -1.2215123283371736879e-289,
  2.6827483411022054912e-241, 3.2988215943776273615e-257, 2.1353977370878701046e-273, -1.2215123283371736879e-289,
  1.1830515272065748694e-241, -3.117427111020820077e-259, -5.9718623963762788119e-275, 6.1155422068568954053e-291,
  4.3320312025875939195e-242, -3.117427111020820077e-259, -5.9718623963762788119e-275, 6.1155422068568954053e-291,
  5.827891678485165325e-243, -3.117427111020820077e-259, -5.9718623963762788119e-275, 6.1155422068568954053e-291,
  5.827891678485165325e-243, -3.117427111020820077e-259, -5.9718623963762788119e-275, 6.1155422068568954053e-291,
  5.827891678485165325e-243, -3.117427111020820077e-259, -5.9718623963762788119e-275, 6.1155422068568954053e-291,
  1.1413391350613183311e-243, -5.1586784110844895013e-260, -1.9524039360882352712e-276, -2.9779654517181717279e-292,
  1.1413391350613183311e-243, -5.1586784110844895013e-260, -1.9524039360882352712e-276, -2.9779654517181717279e-292,
  1.1413391350613183311e-243, -5.1586784110844895013e-260, -1.9524039360882352712e-276, -2.9779654517181717279e-292,
  5.5552006713333735927e-244, 7.8491179384773690214e-260, -1.9524039360882352712e-276, -2.9779654517181717279e-292,
  2.6261053316934700345e-244, 1.345219763696439399e-260, 1.6579848156414234801e-276, 1.0303712682997740506e-292,
  1.1615576618735179302e-244, 1.345219763696439399e-260, 1.6579848156414234801e-276, 1.0303712682997740506e-292,
  4.2928382696354204061e-245, -2.8075477999879273582e-261, -1.472095602234059958e-277, 2.8287088295287585094e-294,
  6.3146909508553973881e-246, 1.2573885592501529789e-261, 3.0408903374280139822e-277, 2.8287088295287585094e-294,
  6.3146909508553973881e-246, 1.2573885592501529789e-261, 3.0408903374280139822e-277, 2.8287088295287585094e-294,
  6.3146909508553973881e-246, 1.2573885592501529789e-261, 3.0408903374280139822e-277, 2.8287088295287585094e-294,
  1.7379794826680480784e-246, 2.4115446944063306384e-262, 2.202741251392177696e-278, 2.8287088295287585094e-294,
  1.7379794826680480784e-246, 2.4115446944063306384e-262, 2.202741251392177696e-278, 2.8287088295287585094e-294,
  5.9380161562121075096e-247, -1.2904053011746964278e-263, 8.7279092175580810531e-280, 8.8634899828990930877e-296,
  2.1712682097791944335e-248, 2.9746046415267896827e-264, -8.6516445844406224413e-282, -5.0528699238150276549e-299,
  2.1712682097791944335e-248, 2.9746046415267896827e-264, -8.6516445844406224413e-282, -5.0528699238150276549e-299,
  2.1712682097791944335e-248, 2.9746046415267896827e-264, -8.6516445844406224413e-282, -5.0528699238150276549e-299,
  2.1712682097791944335e-248, 2.9746046415267896827e-264, -8.6516445844406224413e-282, -5.0528699238150276549e-299,
  2.1712682097791944335e-248, 2.9746046415267896827e-264, -8.6516445844406224413e-282, -5.0528699238150276549e-299,
  3.8349029251851101018e-249, -2.6436684620390282645e-267, -4.3807022524130141006e-284, -2.7456019707854725967e-300,
  3.8349029251851101018e-249, -2.6436684620390282645e-267, -4.3807022524130141006e-284, -2.7456019707854725967e-300,
  3.8349029251851101018e-249, -2.6436684620390282645e-267, -4.3807022524130141006e-284, -2.7456019707854725967e-300,
  1.6001805286092554504e-249, -2.6436684620390282645e-267, -4.3807022524130141006e-284, -2.7456019707854725967e-300,
  4.8281933032132812475e-250, -2.6436684620390282645e-267, -4.3807022524130141006e-284, -2.7456019707854725967e-300,
  4.8281933032132812475e-250, -2.6436684620390282645e-267, -4.3807022524130141006e-284, -2.7456019707854725967e-300,
  2.0347903074934629333e-250, -2.6436684620390282645e-267, -4.3807022524130141006e-284, -2.7456019707854725967e-300,
  6.3808880963355377617e-251, -2.6436684620390282645e-267, -4.3807022524130141006e-284, -2.7456019707854725967e-300,
  6.3808880963355377617e-251, -2.6436684620390282645e-267, -4.3807022524130141006e-284, -2.7456019707854725967e-300,
  2.8891343516857640937e-251, 5.1095823452235464813e-267, -4.3807022524130141006e-284, -2.7456019707854725967e-300,
  1.1432574793608780349e-251, 1.2329569415922591084e-267, -4.3807022524130141006e-284, -2.7456019707854725967e-300,
  2.7031904319843490867e-252, 2.638005906844371576e-268, 6.3790946999826013345e-284, -2.7456019707854725967e-300,
  2.7031904319843490867e-252, 2.638005906844371576e-268, 6.3790946999826013345e-284, -2.7456019707854725967e-300,
  5.2084434157824127104e-253, 2.1511502957481757317e-269, 3.2670891426006735363e-285, 2.4084160842482777461e-301,
  5.2084434157824127104e-253, 2.1511502957481757317e-269, 3.2670891426006735363e-285, 2.4084160842482777461e-301,
  5.2084434157824127104e-253, 2.1511502957481757317e-269, 3.2670891426006735363e-285, 2.4084160842482777461e-301,
  2.4805108027747776379e-253, 2.1511502957481757317e-269, 3.2670891426006735363e-285, 2.4084160842482777461e-301,
  1.1165444962709601017e-253, 2.1511502957481757317e-269, 3.2670891426006735363e-285, 2.4084160842482777461e-301,
  4.3456134301905148502e-254, 6.3684349745470443788e-270, -9.5347405022956030541e-287, -1.5805886663557401565e-302,
  9.3569766393097138822e-255, 2.5826679788133653036e-270, -9.5347405022956030541e-287, -1.5805886663557401565e-302,
  9.3569766393097138822e-255, 2.5826679788133653036e-270, -9.5347405022956030541e-287, -1.5805886663557401565e-302,
  8.3218722366085688343e-256, -2.0046830753539152442e-272, -3.4057806738724185961e-288, 2.3458177946667328156e-304,
  8.3218722366085688343e-256, -2.0046830753539152442e-272, -3.4057806738724185961e-288, 2.3458177946667328156e-304,
  8.3218722366085688343e-256, -2.0046830753539152442e-272, -3.4057806738724185961e-288, 2.3458177946667328156e-304,
  8.3218722366085688343e-256, -2.0046830753539152442e-272, -3.4057806738724185961e-288, 2.3458177946667328156e-304,
  2.9938788518280315834e-256, -2.0046830753539152442e-272, -3.4057806738724185961e-288, 2.3458177946667328156e-304,
  3.2988215943776273615e-257, 2.1353977370878701046e-273, -1.2215123283371736879e-289, 6.7342163555358599277e-306,
  3.2988215943776273615e-257, 2.1353977370878701046e-273, -1.2215123283371736879e-289, 6.7342163555358599277e-306,
  3.2988215943776273615e-257, 2.1353977370878701046e-273, -1.2215123283371736879e-289, 6.7342163555358599277e-306,
  3.2988215943776273615e-257, 2.1353977370878701046e-273, -1.2215123283371736879e-289, 6.7342163555358599277e-306,
  1.6338236616337094706e-257, 2.1353977370878701046e-273, -1.2215123283371736879e-289, 6.7342163555358599277e-306,
  8.0132469526175071002e-258, 2.8687869620228451614e-274, -1.9537812801257956865e-290, 1.0380272777574237546e-306,
  3.850752120757712373e-258, 2.8687869620228451614e-274, -1.9537812801257956865e-290, 1.0380272777574237546e-306,
  1.7695047048278150093e-258, 2.8687869620228451614e-274, -1.9537812801257956865e-290, 1.0380272777574237546e-306,
  7.2888099686286655858e-259, 5.581381609158630475e-275, 6.1155422068568946933e-291, 1.0380272777574237546e-306,
  2.0856914288039227544e-259, -1.9524039360882352712e-276, -2.9779654517181712829e-292, -3.000817432603284506e-308,
  2.0856914288039227544e-259, -1.9524039360882352712e-276, -2.9779654517181712829e-292, -3.000817432603284506e-308,
  7.8491179384773690214e-260, -1.9524039360882352712e-276, -2.9779654517181712829e-292, -3.000817432603284506e-308,
  1.345219763696439399e-260, 1.6579848156414234801e-276, 1.0303712682997738281e-292, 1.4493302844111182601e-308,
  1.345219763696439399e-260, 1.6579848156414234801e-276, 1.0303712682997738281e-292, 1.4493302844111182601e-308,
  1.345219763696439399e-260, 1.6579848156414234801e-276, 1.0303712682997738281e-292, 1.4493302844111182601e-308,
  5.3223249184882342185e-261, -1.472095602234059958e-277, 2.8287088295287585094e-294, -1.0874435234232647519e-310,
  1.2573885592501529789e-261, 3.0408903374280139822e-277, 2.8287088295287585094e-294, -1.0874435234232647519e-310,
  1.2573885592501529789e-261, 3.0408903374280139822e-277, 2.8287088295287585094e-294, -1.0874435234232647519e-310,
  2.4115446944063306384e-262, 2.202741251392177696e-278, 2.8287088295287585094e-294, -1.0874435234232647519e-310,
  2.4115446944063306384e-262, 2.202741251392177696e-278, 2.8287088295287585094e-294, -1.0874435234232647519e-310,
  2.4115446944063306384e-262, 2.202741251392177696e-278, 2.8287088295287585094e-294, -1.0874435234232647519e-310,
  1.1412520821444306741e-262, -6.1787496089661820348e-279, -3.028042329852615431e-295, -2.182740474438892116e-311,
  5.0610577601348040988e-263, 7.9243314524777990283e-279, -3.028042329852615431e-295, -2.182740474438892116e-311,
  1.8853262294800541881e-263, 8.7279092175580810531e-280, 8.8634899828990930877e-296, -9.8167844904532653004e-314,
  2.9746046415267896827e-264, -8.6516445844406224413e-282, -5.0528699238150265939e-299, -1.3288013265921760399e-314,
  2.9746046415267896827e-264, -8.6516445844406224413e-282, -5.0528699238150265939e-299, -1.3288013265921760399e-314,
  2.9746046415267896827e-264, -8.6516445844406224413e-282, -5.0528699238150265939e-299, -1.3288013265921760399e-314,
  9.8977243486757054781e-265, -8.6516445844406224413e-282, -5.0528699238150265939e-299, -1.3288013265921760399e-314,
  9.8977243486757054781e-265, -8.6516445844406224413e-282, -5.0528699238150265939e-299, -1.3288013265921760399e-314,
  4.9356438320276576408e-265, -8.6516445844406224413e-282, -5.0528699238150265939e-299, -1.3288013265921760399e-314,
  2.4546035737036337221e-265, -8.6516445844406224413e-282, -5.0528699238150265939e-299, -1.3288013265921760399e-314,
  1.2140834445416214873e-265, 1.8893435613692150014e-281, 3.0075895258731974416e-297, -9.8167844904532653004e-314,
  5.9382337996061564537e-266, 5.1208955146257653156e-282, -5.0528699238150265939e-299, -1.3288013265921760399e-314,
  2.8369334767011265554e-266, 5.1208955146257653156e-282, -5.0528699238150265939e-299, -1.3288013265921760399e-314,
  1.2862833152486119506e-266, 1.6777604898591683764e-282, -5.0528699238150265939e-299, -1.3288013265921760399e-314,
  5.1095823452235464813e-267, -4.3807022524130141006e-284, -2.7456019707854725967e-300, -2.5539572388808429997e-317,
  1.2329569415922591084e-267, -4.3807022524130141006e-284, -2.7456019707854725967e-300, -2.5539572388808429997e-317,
  1.2329569415922591084e-267, -4.3807022524130141006e-284, -2.7456019707854725967e-300, -2.5539572388808429997e-317,
  2.638005906844371576e-268, 6.3790946999826013345e-284, -2.7456019707854725967e-300, -2.5539572388808429997e-317,
  2.638005906844371576e-268, 6.3790946999826013345e-284, -2.7456019707854725967e-300, -2.5539572388808429997e-317,
  2.1511502957481757317e-269, 3.2670891426006735363e-285, 2.4084160842482773317e-301, 5.7350888195772519812e-317,
  2.1511502957481757317e-269, 3.2670891426006735363e-285, 2.4084160842482773317e-301, 5.7350888195772519812e-317,
  2.1511502957481757317e-269, 3.2670891426006735363e-285, 2.4084160842482773317e-301, 5.7350888195772519812e-317,
  2.1511502957481757317e-269, 3.2670891426006735363e-285, 2.4084160842482773317e-301, 5.7350888195772519812e-317,
  6.3684349745470443788e-270, -9.5347405022956030541e-287, -1.5805886663557401565e-302, 3.6369654387311681856e-319,
  6.3684349745470443788e-270, -9.5347405022956030541e-287, -1.5805886663557401565e-302, 3.6369654387311681856e-319,
  2.5826679788133653036e-270, -9.5347405022956030541e-287, -1.5805886663557401565e-302, 3.6369654387311681856e-319,
  6.8978448094652555593e-271, 1.1480487920352081009e-286, 7.5257037990230704094e-303, 3.6369654387311681856e-319,
  6.8978448094652555593e-271, 1.1480487920352081009e-286, 7.5257037990230704094e-303, 3.6369654387311681856e-319,
  2.1656360647981577662e-271, 9.7287370902823839435e-288, 1.6928061833779524157e-303, 3.6369654387311681856e-319,
  2.1656360647981577662e-271, 9.7287370902823839435e-288, 1.6928061833779524157e-303, 3.6369654387311681856e-319,
  9.825838786313830552e-272, 9.7287370902823839435e-288, 1.6928061833779524157e-303, 3.6369654387311681856e-319,
  3.9105778554799569972e-272, 9.7287370902823839435e-288, 1.6928061833779524157e-303, 3.6369654387311681856e-319,
  9.5294739006302120482e-273, -1.2215123283371736879e-289, 6.7342163555358599277e-306, -5.681754927174335258e-322,
  9.5294739006302120482e-273, -1.2215123283371736879e-289, 6.7342163555358599277e-306, -5.681754927174335258e-322,
  2.1353977370878701046e-273, -1.2215123283371736879e-289, 6.7342163555358599277e-306, -5.681754927174335258e-322,
  2.1353977370878701046e-273, -1.2215123283371736879e-289, 6.7342163555358599277e-306, -5.681754927174335258e-322,
  2.8687869620228451614e-274, -1.9537812801257956865e-290, 1.0380272777574237546e-306, 6.4228533959362050743e-323,
};
 __attribute__((aligned(64)))
const float PayneHanekReductionTable_float[] = {
    // clang-format off
  0.159154892, 5.112411827e-08, 3.626141271e-15, -2.036222915e-22,
  0.03415493667, 6.420638243e-09, 7.342738037e-17, 8.135951656e-24,
  0.03415493667, 6.420638243e-09, 7.342738037e-17, 8.135951656e-24,
  0.002904943191, -9.861969574e-11, -9.839336547e-18, -1.790215892e-24,
  0.002904943191, -9.861969574e-11, -9.839336547e-18, -1.790215892e-24,
  0.002904943191, -9.861969574e-11, -9.839336547e-18, -1.790215892e-24,
  0.002904943191, -9.861969574e-11, -9.839336547e-18, -1.790215892e-24,
  0.0009518179577, 1.342109202e-10, 1.791623576e-17, 1.518506657e-24,
  0.0009518179577, 1.342109202e-10, 1.791623576e-17, 1.518506657e-24,
  0.0004635368241, 1.779561221e-11, 4.038449606e-18, -1.358546052e-25,
  0.0002193961991, 1.779561221e-11, 4.038449606e-18, -1.358546052e-25,
  9.73258866e-05, 1.779561221e-11, 4.038449606e-18, -1.358546052e-25,
  3.62907449e-05, 3.243700447e-12, 5.690024473e-19, 7.09405479e-26,
  5.773168596e-06, 1.424711477e-12, 1.3532163e-19, 1.92417627e-26,
  5.773168596e-06, 1.424711477e-12, 1.3532163e-19, 1.92417627e-26,
  5.773168596e-06, 1.424711477e-12, 1.3532163e-19, 1.92417627e-26,
  1.958472239e-06, 5.152167755e-13, 1.3532163e-19, 1.92417627e-26,
  5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
  5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
  5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
  5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
  5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
  5.112411827e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
  2.132179588e-08, 3.626141271e-15, -2.036222915e-22, 6.177847236e-30,
  6.420638243e-09, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
  6.420638243e-09, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
  2.695347945e-09, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
  8.327027956e-10, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
  8.327027956e-10, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
  3.670415083e-10, 7.342738037e-17, 8.135951656e-24, -1.330400526e-31,
  1.342109202e-10, 1.791623576e-17, 1.518506361e-24, 2.613904e-31,
  1.779561221e-11, 4.038449606e-18, -1.358545683e-25, -3.443243946e-32,
  1.779561221e-11, 4.038449606e-18, -1.358545683e-25, -3.443243946e-32,
  1.779561221e-11, 4.038449606e-18, -1.358545683e-25, -3.443243946e-32,
  3.243700447e-12, 5.690024473e-19, 7.094053557e-26, 1.487136711e-32,
  3.243700447e-12, 5.690024473e-19, 7.094053557e-26, 1.487136711e-32,
  3.243700447e-12, 5.690024473e-19, 7.094053557e-26, 1.487136711e-32,
  1.424711477e-12, 1.3532163e-19, 1.924175961e-26, 2.545416018e-33,
  5.152167755e-13, 1.3532163e-19, 1.924175961e-26, 2.545416018e-33,
  6.046956013e-14, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
  6.046956013e-14, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
  6.046956013e-14, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
  3.626141271e-15, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
  3.626141271e-15, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
  3.626141271e-15, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
  3.626141271e-15, -2.036222915e-22, 6.177846108e-30, 1.082084378e-36,
  7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
  7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
  7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
  7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
  7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
  7.342738037e-17, 8.135951656e-24, -1.330400526e-31, 6.296048013e-40,
  1.791623576e-17, 1.518506361e-24, 2.61390353e-31, 4.764937743e-38,
  1.791623576e-17, 1.518506361e-24, 2.61390353e-31, 4.764937743e-38,
  4.038449606e-18, -1.358545683e-25, -3.443243946e-32, 6.296048013e-40,
  4.038449606e-18, -1.358545683e-25, -3.443243946e-32, 6.296048013e-40,
  5.690024473e-19, 7.094053557e-26, 1.487136711e-32, 6.296048013e-40,
  5.690024473e-19, 7.094053557e-26, 1.487136711e-32, 6.296048013e-40,
  5.690024473e-19, 7.094053557e-26, 1.487136711e-32, 6.296048013e-40,
  1.3532163e-19, 1.924175961e-26, 2.545415467e-33, 6.296048013e-40,
  1.3532163e-19, 1.924175961e-26, 2.545415467e-33, 6.296048013e-40,
  2.690143217e-20, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
  2.690143217e-20, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
  2.690143217e-20, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
  1.334890502e-20, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
  6.572641438e-21, -1.452834402e-28, -6.441077673e-36, -1.764234767e-42,
  0.05874381959, 1.222115387e-08, 7.693612965e-16, 1.792054435e-22,
  0.02749382704, 4.77057327e-09, 7.693612965e-16, 1.792054435e-22,
  0.01186883077, 1.045283415e-09, 3.252721926e-16, 7.332633139e-23,
  0.00405633077, 1.045283415e-09, 3.252721926e-16, 7.332633139e-23,
  0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
  0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
  0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
  0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
  0.000150081818, -2.454155802e-12, 1.161414894e-20, 1.291319272e-27,
  2.801149822e-05, 4.821800945e-12, 8.789757674e-19, 1.208447639e-25,
  2.801149822e-05, 4.821800945e-12, 8.789757674e-19, 1.208447639e-25,
  2.801149822e-05, 4.821800945e-12, 8.789757674e-19, 1.208447639e-25,
  1.275271279e-05, 1.183823005e-12, 1.161414894e-20, 1.291319272e-27,
  5.12331826e-06, 1.183823005e-12, 1.161414894e-20, 1.291319272e-27,
  1.308621904e-06, 2.743283031e-13, 1.161414894e-20, 1.291319272e-27,
  1.308621904e-06, 2.743283031e-13, 1.161414894e-20, 1.291319272e-27,
  3.549478151e-07, 4.695462769e-14, 1.161414894e-20, 1.291319272e-27,
  3.549478151e-07, 4.695462769e-14, 1.161414894e-20, 1.291319272e-27,
  1.165292645e-07, 1.853292503e-14, 4.837885366e-21, 1.291319272e-27,
  1.165292645e-07, 1.853292503e-14, 4.837885366e-21, 1.291319272e-27,
  5.69246339e-08, 4.322073705e-15, 1.449754789e-21, 7.962890365e-29,
  2.712231151e-08, 4.322073705e-15, 1.449754789e-21, 7.962890365e-29,
  1.222115387e-08, 7.693612965e-16, 1.792054182e-22, 2.91418027e-29,
  4.77057327e-09, 7.693612965e-16, 1.792054182e-22, 2.91418027e-29,
  1.045283415e-09, 3.252721926e-16, 7.332632508e-23, 3.898253736e-30,
  1.045283415e-09, 3.252721926e-16, 7.332632508e-23, 3.898253736e-30,
  1.139611461e-10, 1.996093359e-17, 5.344349223e-25, 1.511644828e-31,
  1.139611461e-10, 1.996093359e-17, 5.344349223e-25, 1.511644828e-31,
  1.139611461e-10, 1.996093359e-17, 5.344349223e-25, 1.511644828e-31,
  1.139611461e-10, 1.996093359e-17, 5.344349223e-25, 1.511644828e-31,
  5.575349904e-11, 6.083145782e-18, 5.344349223e-25, 1.511644828e-31,
  2.664967552e-11, -8.557475018e-19, -8.595036458e-26, -2.139883875e-32,
  1.209775682e-11, 2.61369883e-18, 5.344349223e-25, 1.511644828e-31,
  4.821800945e-12, 8.789757674e-19, 1.208447639e-25, 3.253064536e-33,
  1.183823005e-12, 1.161414894e-20, 1.29131908e-27, 1.715766248e-34,
  1.183823005e-12, 1.161414894e-20, 1.29131908e-27, 1.715766248e-34,
  2.743283031e-13, 1.161414894e-20, 1.29131908e-27, 1.715766248e-34,
    // clang-format on
};
#endif // HWY_ONCE

