
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
constexpr float TrigRangeMax = 39000; // Max value for using 4-part sum of Pi
constexpr float PiA2f = 3.1414794921875f; // Three-part sum of Pi (1/3)
constexpr float PiB2f = 0.00011315941810607910156f; // Three-part sum of Pi (2/3)
constexpr float PiC2f = 1.9841872589410058936e-09f; // Three-part sum of Pi (3/3)
constexpr float TrigRangeMax2 = 125.0f; // Max value for using 3-part sum of Pi
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

// Normalizes a double-float precision representation (redistributes hi vs. lo value)
// Translated from common/df.h:102 dfnormalize_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) NormalizeDF(const D df, Vec2<D> t) {
  Vec<D> s = Add(Get2<0>(t), Get2<1>(t));
  return Create2(df, s, Add(Sub(Get2<0>(t), s), Get2<1>(t)));
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

// Computes x + y in double-float precision
// Translated from common/df.h:158 dfadd2_vf2_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) AddDF(const D df, Vec2<D> x, Vec2<D> y) {
  Vec<D> s = Add(Get2<0>(x), Get2<0>(y));
  Vec<D> v = Sub(s, Get2<0>(x));
  Vec<D> t = Add(Sub(Get2<0>(x), Sub(s, v)), Sub(Get2<0>(y), v));
  return Create2(df, s, Add(t, Add(Get2<1>(x), Get2<1>(y))));
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

// Computes x + y in double-float precision
// Translated from common/df.h:116 dfadd2_vf2_vf_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) AddDF(const D df, Vec<D> x, Vec<D> y) {
  Vec<D> s = Add(x, y);
  Vec<D> v = Sub(s, x);
  return Create2(df, s, Add(Sub(x, Sub(s, v)), Sub(y, v)));
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

// Computes x * y in double-float precision
// Translated from common/df.h:107 dfscale_vf2_vf2_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) ScaleDF(const D df, Vec2<D> d, Vec<D> s) {
  return Create2(df, Mul(Get2<0>(d), s), Mul(Get2<1>(d), s));
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

// Sets the exponent of 'x' to 2^e. Very fast, "no denormal"
// Translated from libm/sleefsimdsp.c:539 vldexp3_vf_vf_vi2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) LoadExp3(const D df, Vec<D> d, Vec<RebindToSigned<D>> q) {
  RebindToSigned<D> di;
  
  return BitCast(df, Add(BitCast(di, d), ShiftLeft<23>(q)));
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

// Normalizes a double-double precision representation (redistributes hi vs. lo value)
// Translated from common/dd.h:108 ddnormalize_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) NormalizeDD(const D df, Vec2<D> t) {
  Vec<D> s = Add(Get2<0>(t), Get2<1>(t));
  return Create2(df, s, Add(Sub(Get2<0>(t), s), Get2<1>(t)));
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

// Computes x + y in double-double precision, sped up by assuming |x| > |y|
// Translated from common/dd.h:159 ddadd_vd2_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) AddFastDD(const D df, Vec2<D> x, Vec2<D> y) {
  // |x| >= |y|

  Vec<D> s = Add(Get2<0>(x), Get2<0>(y));
  return Create2(df, s, Add4(df, Sub(Get2<0>(x), s), Get2<0>(y), Get2<1>(x), Get2<1>(y)));
}

// Add ((((v0 + 1) + v2) + v3) + v4) + v5
// Translated from common/dd.h:71 vadd_vd_6vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Add6(const D df, Vec<D> v0, Vec<D> v1, Vec<D> v2, Vec<D> v3, Vec<D> v4, Vec<D> v5) {
  return Add5(df, Add(v0, v1), v2, v3, v4, v5);
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

// Computes x + y in double-double precision
// Translated from common/dd.h:124 ddadd2_vd2_vd_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) AddDD(const D df, Vec<D> x, Vec<D> y) {
  Vec<D> s = Add(x, y);
  Vec<D> v = Sub(s, x);
  return Create2(df, s, Add(Sub(x, Sub(s, v)), Sub(y, v)));
}

// Computes x + y in double-double precision, sped up by assuming |x| > |y|
// Translated from common/dd.h:130 ddadd_vd2_vd2_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) AddFastDD(const D df, Vec2<D> x, Vec<D> y) {
  Vec<D> s = Add(Get2<0>(x), y);
  return Create2(df, s, Add3(df, Sub(Get2<0>(x), s), y, Get2<1>(x)));
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

// Computes x * y in double-double precision
// Translated from common/dd.h:113 ddscale_vd2_vd2_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) ScaleDD(const D df, Vec2<D> d, Vec<D> s) {
  return Create2(df, Mul(Get2<0>(d), s), Mul(Get2<1>(d), s));
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

// Sets the exponent of 'x' to 2^e. Very fast, "no denormal"
// Translated from common/commonfuncs.h:353 vldexp3_vd_vd_vi
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) LoadExp3(const D df, Vec<D> d, Vec<RebindToSigned<D>> q) {
  RebindToUnsigned<D> du;
  
  return BitCast(df, Add(BitCast(du, d), ShiftLeft<32>(BitCast(du, ShiftLeft<20>(q)))));
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

// Bitwise or of x with sign bit of y
// Translated from common/commonfuncs.h:224 vorsign_vd_vd_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) OrSignBit(const D df, Vec<D> x, Vec<D> y) {
  RebindToUnsigned<D> du;
  
  return BitCast(df, Or(BitCast(du, x), SignBit(df, y)));
}

// True if d is an integer
// Translated from common/commonfuncs.h:278 visint_vo_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Mask<D>) IsInt(const D df, Vec<D> d) {
  return Eq(Round(d), d);
}

// True if d is an odd (assuming d is an integer)
// Translated from common/commonfuncs.h:282 visodd_vo_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Mask<D>) IsOdd(const D df, Vec<D> d) {
  Vec<D> x = Mul(d, Set(df, 0.5));
  return Ne(Round(x), x);
}

// Create a mask of which is true if x's sign bit is set
// Translated from common/commonfuncs.h:200 vsignbit_vo_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Mask<D>) SignBitMask(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  
  return RebindMask(df, Eq(And(BitCast(du, d), BitCast(du, Set(df, -0.0))), BitCast(du, Set(df, -0.0))));
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
  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2));

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
  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2));

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
  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2));

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
  Mask<D> g = Lt(Abs(r), Set(df, TrigRangeMax2));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    s = ConvertTo(df, q);
    u = MulAdd(s, Set(df, -PiAf), r);
    u = MulAdd(s, Set(df, -PiBf), u);
    u = MulAdd(s, Set(df, -PiCf), u);
    u = MulAdd(s, Set(df, -PiDf), u);

    d = IfThenElse(g, d, u);
    g = Lt(Abs(r), Set(df, TrigRangeMax));

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
  Mask<D> g = Lt(Abs(r), Set(df, TrigRangeMax2));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    s = ConvertTo(df, q);
    u = MulAdd(s, Set(df, -PiAf*0.5f), r);
    u = MulAdd(s, Set(df, -PiBf*0.5f), u);
    u = MulAdd(s, Set(df, -PiCf*0.5f), u);
    u = MulAdd(s, Set(df, -PiDf*0.5f), u);

    d = IfThenElse(g, d, u);
    g = Lt(Abs(r), Set(df, TrigRangeMax));

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
  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2*0.5f));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    Vec<RebindToSigned<D>> q2 = NearestInt(Mul(d, Set(df, (float)(2 * OneOverPi))));
    s = ConvertTo(df, q);
    u = MulAdd(s, Set(df, -PiAf*0.5f), d);
    u = MulAdd(s, Set(df, -PiBf*0.5f), u);
    u = MulAdd(s, Set(df, -PiCf*0.5f), u);
    u = MulAdd(s, Set(df, -PiDf*0.5f), u);

    q = IfThenElse(RebindMask(di, g), q, q2);
    x = IfThenElse(g, x, u);
    g = Lt(Abs(d), Set(df, TrigRangeMax));

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

}  // namespace sleef
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_SLEEF_SLEEF_INL_

#if HWY_ONCE
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

