
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
#include "Estrin.h"
#include "AVX512FloatUtils.h"

extern const float PayneHanekReductionTable_float[];
extern const double PayneHanekReductionTable_double[];

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
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

// Computes sqrt(x) with 0.5001 ULP accuracy
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

// Computes sqrt(x^2 + y^2) with 0.5001 ULP accuracy
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

// Computes sin(x) and cos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1360 XSINCOSF_U1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SinCos(const D df, Vec<D> d);

// Computes sin(x) and cos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1233 XSINCOSF
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SinCosFast(const D df, Vec<D> d);

// Computes sin(x*pi) and cos(x*pi) with max(0.506 ULP, FLT_MIN) accuracy (in practice staying below 2 ULP even for subnormals)
// Translated from libm/sleefsimdsp.c:1477 XSINCOSPIF_U05
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SinCosPi(const D df, Vec<D> d);

// Computes sin(x*pi) and cos(x*pi) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1537 XSINCOSPIF_U35
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SinCosPiFast(const D df, Vec<D> d);

// Computes sin(x*pi) with max(0.506 ULP, FLT_MIN) accuracy (in practice staying below 2 ULP even for subnormals)
// Translated from libm/sleefsimdsp.c:3289 xsinpif_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) SinPi(const D df, Vec<D> d);

// Computes cos(x*pi) with max(0.506 ULP, FLT_MIN) accuracy (in practice staying below 2 ULP even for subnormals)
// Translated from libm/sleefsimdsp.c:3339 xcospif_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) CosPi(const D df, Vec<D> d);

// Computes acos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1948 xacosf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Acos(const D df, Vec<D> d);

// Computes asin(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1928 xasinf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Asin(const D df, Vec<D> d);

// Computes asin(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1831 xasinf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) AsinFast(const D df, Vec<D> d);

// Computes acos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1847 xacosf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) AcosFast(const D df, Vec<D> d);

// Computes atan(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1743 xatanf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) AtanFast(const D df, Vec<D> d);

// Computes atan(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1973 xatanf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Atan(const D df, Vec<D> d);

// Computes atan(y/x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1911 xatan2f_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Atan2(const D df, Vec<D> y, Vec<D> x);

// Computes atan(y/x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1818 xatan2f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Atan2Fast(const D df, Vec<D> y, Vec<D> x);

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

// Computes asinh(x) with 1 ULP accuracy
// Translated from libm/sleefsimdsp.c:2554 xasinhf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Asinh(const D df, Vec<D> x);

// Computes acosh(x) with 1 ULP accuracy
// Translated from libm/sleefsimdsp.c:2575 xacoshf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Acosh(const D df, Vec<D> x);

// Computes atanh(x) with 1 ULP accuracy
// Translated from libm/sleefsimdsp.c:2591 xatanhf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Atanh(const D df, Vec<D> x);

// Computes erf(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:3471 xerff_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Erf(const D df, Vec<D> a);

// Computes 1 - erf(x) with 1.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:3536 xerfcf_u15
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Erfc(const D df, Vec<D> a);

// Computes gamma(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:3428 xtgammaf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Gamma(const D df, Vec<D> a);

// Computes log(gamma(x)) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:3447 xlgammaf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) LogGamma(const D df, Vec<D> a);

// Computes fmod(x), the floating point remainder
// Translated from libm/sleefsimdsp.c:3173 xfmodf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Fmod(const D df, Vec<D> x, Vec<D> y);

// Computes remainder(x), the signed floating point remainder
// Translated from libm/sleefsimdsp.c:3218 xremainderf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Remainder(const D df, Vec<D> x, Vec<D> y);

// Computes x * 2^exp
// Translated from libm/sleefsimdsp.c:543 xldexpf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) LdExp(const D df, Vec<D> x, Vec<RebindToSigned<D>> q);

// Decomposes x into 2^exp * fr where abs(fr) is in [0.5, 1), returning fr
// Translated from libm/sleefsimdsp.c:3128 xfrfrexpf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) FrFrexp(const D df, Vec<D> x);

// Decomposes x into 2^exp * fr where abs(fr) is in [0.5, 1), returning exp
// Translated from libm/sleefsimdsp.c:3144 xexpfrexpf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<RebindToSigned<D>>) ExpFrexp(const D df, Vec<D> x);

// Computes the unbiased exponent of x
// Translated from libm/sleefsimdsp.c:508 xilogbf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<RebindToSigned<D>>) ILogB(const D df, Vec<D> d);

// Decompose x into an integer and fractional part
// Translated from libm/sleefsimdsp.c:1591 XMODFF
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) Modf(const D df, Vec<D> x);

// Returns the next representable value after x in the direction of y
// Translated from libm/sleefsimdsp.c:3105 xnextafterf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) NextAfter(const D df, Vec<D> x, Vec<D> y);

// Computes sin(pi) very quickly in range [-30, 30] with max(2e-6, 350 ULP) accuracy. Falls back to SinFast out of range.
// Translated from libm/sleefsimdsp.c:1165 xfastsinf_u3500
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) SinFaster(const D df, Vec<D> d);

// Computes cos(pi) very quickly in range [-30, 30] with max(2e-6, 350 ULP) accuracy. Falls back to CosFast out of range.
// Translated from libm/sleefsimdsp.c:1189 xfastcosf_u3500
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) CosFaster(const D df, Vec<D> d);

// Computes x^y very quickly with 350 ULP accuracy
// Translated from libm/sleefsimdsp.c:2403 xfastpowf_u3500
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) PowFaster(const D df, Vec<D> x, Vec<D> y);

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

// Computes sqrt(x) with 0.5001 ULP accuracy
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

// Computes sqrt(x^2 + y^2) with 0.5001 ULP accuracy
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

// Computes acos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2007 xacos_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Acos(const D df, Vec<D> d);

// Computes asin(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:1945 xasin_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Asin(const D df, Vec<D> d);

// Computes atan(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2042 xatan_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Atan(const D df, Vec<D> d);

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

// Computes atan(y/x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:1902 xatan2_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Atan2(const D df, Vec<D> y, Vec<D> x);

// Computes atan(y/x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:1890 xatan2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Atan2Fast(const D df, Vec<D> y, Vec<D> x);

// Computes sin(x) and cos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:1086 XSINCOS_U1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SinCos(const D df, Vec<D> d);

// Computes sin(x) and cos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:942 XSINCOS
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SinCosFast(const D df, Vec<D> d);

// Computes sin(x*pi) and cos(x*pi) with max(0.506 ULP, FLT_MIN) accuracy (in practice staying below 2 ULP even for subnormals)
// Translated from libm/sleefsimddp.c:1245 XSINCOSPI_U05
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SinCosPi(const D df, Vec<D> d);

// Computes sin(x*pi) and cos(x*pi) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:1311 XSINCOSPI_U35
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SinCosPiFast(const D df, Vec<D> d);

// Computes sin(x*pi) with max(0.506 ULP, DBL_MIN) accuracy (in practice staying below 2 ULP even for subnormals)
// Translated from libm/sleefsimddp.c:1456 xsinpi_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) SinPi(const D df, Vec<D> d);

// Computes cos(x*pi) with max(0.506 ULP, DBL_MIN) accuracy (in practice staying below 2 ULP even for subnormals)
// Translated from libm/sleefsimddp.c:1507 xcospi_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) CosPi(const D df, Vec<D> d);

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

// Computes asinh(x) with 1 ULP accuracy
// Translated from libm/sleefsimddp.c:2539 xasinh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Asinh(const D df, Vec<D> x);

// Computes acosh(x) with 1 ULP accuracy
// Translated from libm/sleefsimddp.c:2561 xacosh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Acosh(const D df, Vec<D> x);

// Computes atanh(x) with 1 ULP accuracy
// Translated from libm/sleefsimddp.c:2576 xatanh
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Atanh(const D df, Vec<D> x);

// Computes erf(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:3470 xerf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Erf(const D df, Vec<D> a);

// Computes 1 - erf(x) with 1.5 ULP accuracy
// Translated from libm/sleefsimddp.c:3565 xerfc_u15
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Erfc(const D df, Vec<D> a);

// Computes gamma(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:3427 xtgamma_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Gamma(const D df, Vec<D> a);

// Computes log(gamma(x)) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:3446 xlgamma_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) LogGamma(const D df, Vec<D> a);

// Computes fmod(x), the floating point remainder
// Translated from libm/sleefsimddp.c:3269 xfmod
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Fmod(const D df, Vec<D> x, Vec<D> y);

// Computes remainder(x), the signed floating point remainder
// Translated from libm/sleefsimddp.c:3314 xremainder
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Remainder(const D df, Vec<D> x, Vec<D> y);

// Computes x * 2^exp
// Translated from libm/sleefsimddp.c:338 xldexp
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) LdExp(const D df, Vec<D> x, Vec<RebindToSigned<D>> q);

// Decomposes x into 2^exp * fr where abs(fr) is in [0.5, 1), returning fr
// Translated from libm/sleefsimddp.c:3079 xfrfrexp
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) FrFrexp(const D df, Vec<D> x);

// Decomposes x into 2^exp * fr where abs(fr) is in [0.5, 1), returning exp
// Translated from libm/sleefsimddp.c:3094 xexpfrexp
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<RebindToSigned<D>>) ExpFrexp(const D df, Vec<D> x);

// Computes the unbiased exponent of x
// Translated from libm/sleefsimddp.c:340 xilogb
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<RebindToSigned<D>>) ILogB(const D df, Vec<D> d);

// Decompose x into an integer and fractional part
// Translated from libm/sleefsimddp.c:1371 XMODF
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) Modf(const D df, Vec<D> x);

// Returns the next representable value after x in the direction of y
// Translated from libm/sleefsimddp.c:3056 xnextafter
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) NextAfter(const D df, Vec<D> x, Vec<D> y);

namespace {

//////////////////
// Constants
//////////////////
constexpr double Pi = 3.141592653589793238462643383279502884; // pi
constexpr double OneOverPi = 0.318309886183790671537767526745028724; // 1 / pi
constexpr double TwoOverPi = 0.636619772367581343075535053490057448; // 2 / pi
constexpr int ILogB0 = ((int)0x80000000); // ilogb(0)
constexpr int ILogBNan = ((int)2147483647); // ilogb(nan)
constexpr float FloatMin = 0x1p-126; // Minimum normal float value
constexpr double DoubleMin = 0x1p-1022; // Minimum normal double value
constexpr int IntMax = 2147483647; // maximum 32-bit int
constexpr double PiA = 3.1415926218032836914; // Four-part sum of Pi (1/4)
constexpr double PiB = 3.1786509424591713469e-08; // Four-part sum of Pi (2/4)
constexpr double PiC = 1.2246467864107188502e-16; // Four-part sum of Pi (3/4)
constexpr double PiD = 1.2736634327021899816e-24; // Four-part sum of Pi (4/4)
constexpr double TrigRangeMax = 1e+14; // Max value for using 4-part sum of Pi
constexpr double PiA2 = 3.141592653589793116; // Three-part sum of Pi (1/3)
constexpr double PiB2 = 1.2246467991473532072e-16; // Three-part sum of Pi (2/3)
constexpr double TrigRangeMax2 = 15; // Max value for using 3-part sum of Pi
constexpr double TwoOverPiHi = 0.63661977236758138243; // TwoOverPiHi + TwoOverPiLo ~= 2 / pi
constexpr double TwoOverPiLo = -3.9357353350364971764e-17; // TwoOverPiHi + TwoOverPiLo ~= 2 / pi
constexpr double SqrtDoubleMax = 1.3407807929942596355e+154; // Square root of max double
constexpr double TrigRangeMax3 = 1e+9; // Cutoff value for cospi and sinpi
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
constexpr float TrigRangeMax4f = 8e+6f; // Cutoff value for cospi and sinpi
constexpr float SqrtFloatMax = 18446743523953729536.0; // Square root of max float
constexpr float Ln2Hi_f = 0.693145751953125f; // Ln2Hi + Ln2Lo ~= ln(2)
constexpr float Ln2Lo_f = 1.428606765330187045e-06f; // Ln2Hi + Ln2Lo ~= ln(2)
constexpr float OneOverLn2_f = 1.442695040888963407359924681001892137426645954152985934135449406931f; // 1 / ln(2)
constexpr float Pif = ((float)Pi); // pi (float)
#if (defined (__GNUC__) || defined (__clang__) || defined(__INTEL_COMPILER)) && !defined(_MSC_VER)
constexpr double NanDouble = __builtin_nan(""); // Double precision NaN
constexpr float NanFloat = __builtin_nanf(""); // Floating point NaN
constexpr double InfDouble = __builtin_inf(); // Double precision infinity
constexpr float InfFloat = __builtin_inff(); // Floating point infinity
#elif defined(_MSC_VER) 
constexpr double InfDouble = (1e+300 * 1e+300); // Double precision infinity
constexpr double NanDouble = (InfDouble - InfDouble); // Double precision NaN
constexpr float InfFloat = ((float)InfDouble); // Floating point infinity
constexpr float NanFloat = ((float)NanDouble); // Floating point NaN
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

// Integer log of x (helper, not top-level)
// Translated from libm/sleefsimdsp.c:489 vilogbk_vi2_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<RebindToSigned<D>>) ILogB1(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Mask<D> o = Lt(d, Set(df, 5.421010862427522E-20f));
  d = IfThenElse(o, Mul(Set(df, 1.8446744073709552E19f), d), d);
  Vec<RebindToSigned<D>> q = And(BitCast(di, ShiftRight<23>(BitCast(du, d))), Set(di, 0xff));
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

// Computes x * y in double-float precision, returning result as float
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
  Vec<RebindToSigned<D>> q = IfThenElseZero(RebindMask(di, Gt(ex, Set(di, 90-25))), Set(di, -64));
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

// Computes x^2 in double-float precision, returning result as float
// Translated from common/df.h:201 dfsqu_vf_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) SquareDF_float(const D df, Vec2<D> x) {
#if HWY_SLEEF_HAS_FMA
  return MulAdd(Get2<0>(x), Get2<0>(x), Add(Mul(Get2<0>(x), Get2<1>(x)), Mul(Get2<0>(x), Get2<1>(x))));
#else
  Vec<D> xh = LowerPrecision(df, Get2<0>(x)), xl = Sub(Get2<0>(x), xh);

  return Add5(df, Mul(xh, Get2<1>(x)), Mul(xh, Get2<1>(x)), Mul(xl, xl), Add(Mul(xh, xl), Mul(xh, xl)), Mul(xh, xh));
#endif
}

// o ? (x1, y1) : (x0, y0)
// Translated from common/df.h:42 vsel_vf2_vo_f_f_f_f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) IfThenElse(const D df, Mask<D> o, float x1, float y1, float x0, float y0) {
  return Create2(df, IfThenElse(o, Set(df, x1), Set(df, x0)), IfThenElse(o, Set(df, y1), Set(df, y0)));
}

// Calculate SinPi without handling special cases in double-float precision
// Translated from libm/sleefsimdsp.c:3251 sinpifk
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SinPiDF(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Mask<D> o;
  Vec<D> u, s, t;
  Vec2<D> x, s2;

  u = Mul(d, Set(df, 4.0));
  Vec<RebindToSigned<D>> q = ConvertTo(di, u);
  q = And(Add(q, Xor(BitCast(di, ShiftRight<31>(BitCast(du, q))), Set(di, 1))), Set(di, ~1));
  o = RebindMask(df, Eq(And(q, Set(di, 2)), Set(di, 2)));

  s = Sub(u, ConvertTo(df, q));
  t = s;
  s = Mul(s, s);
  s2 = MulDF(df, t, t);

  

  u = IfThenElse(o, Set(df, -0.2430611801e-7f), Set(df, +0.3093842054e-6f));
  u = MulAdd(u, s, IfThenElse(o, Set(df, +0.3590577080e-5f), Set(df, -0.3657307388e-4f)));
  u = MulAdd(u, s, IfThenElse(o, Set(df, -0.3259917721e-3f), Set(df, +0.2490393585e-2f)));
  x = AddDF(df, Mul(u, s),
			IfThenElse(df, o, 0.015854343771934509277, 4.4940051354032242811e-10,
					    -0.080745510756969451904, -1.3373665339076936258e-09));
  x = AddDF(df, MulDF(df, s2, x),
			 IfThenElse(df, o, -0.30842512845993041992, -9.0728339030733922277e-09,
					     0.78539818525314331055, -2.1857338617566484855e-08));

  x = MulDF(df, x, IfThenElse(df, o, s2, Create2(df, t, Set(df, 0))));
  x = IfThenElse(df, o, AddDF(df, x, Set(df, 1)), x);

  o = RebindMask(df, Eq(And(q, Set(di, 4)), Set(di, 4)));
  x = Set2<0>(x, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<0>(x)))));
  x = Set2<1>(x, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<1>(x)))));

  return x;
}

// Computes cos(x*pi) in double-float precision
// Translated from libm/sleefsimdsp.c:3301 cospifk
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) CosPiDF(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Mask<D> o;
  Vec<D> u, s, t;
  Vec2<D> x, s2;

  u = Mul(d, Set(df, 4.0));
  Vec<RebindToSigned<D>> q = ConvertTo(di, u);
  q = And(Add(q, Xor(BitCast(di, ShiftRight<31>(BitCast(du, q))), Set(di, 1))), Set(di, ~1));
  o = RebindMask(df, Eq(And(q, Set(di, 2)), Set(di, 0)));

  s = Sub(u, ConvertTo(df, q));
  t = s;
  s = Mul(s, s);
  s2 = MulDF(df, t, t);
  
  

  u = IfThenElse(o, Set(df, -0.2430611801e-7f), Set(df, +0.3093842054e-6f));
  u = MulAdd(u, s, IfThenElse(o, Set(df, +0.3590577080e-5f), Set(df, -0.3657307388e-4f)));
  u = MulAdd(u, s, IfThenElse(o, Set(df, -0.3259917721e-3f), Set(df, +0.2490393585e-2f)));
  x = AddDF(df, Mul(u, s),
			IfThenElse(df, o, 0.015854343771934509277, 4.4940051354032242811e-10,
					    -0.080745510756969451904, -1.3373665339076936258e-09));
  x = AddDF(df, MulDF(df, s2, x),
			 IfThenElse(df, o, -0.30842512845993041992, -9.0728339030733922277e-09,
					     0.78539818525314331055, -2.1857338617566484855e-08));

  x = MulDF(df, x, IfThenElse(df, o, s2, Create2(df, t, Set(df, 0))));
  x = IfThenElse(df, o, AddDF(df, x, Set(df, 1)), x);

  o = RebindMask(df, Eq(And(Add(q, Set(di, 2)), Set(di, 4)), Set(di, 4)));
  x = Set2<0>(x, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<0>(x)))));
  x = Set2<1>(x, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<1>(x)))));

  return x;
}

// Computes sqrt(x) in double-float precision
// Translated from common/df.h:366 dfsqrt_vf2_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SqrtDF(const D df, Vec<D> d) {
  Vec<D> t = Sqrt(d);
  return ScaleDF(df, MulDF(df, AddDF(df, d, MulDF(df, t, t)), RecDF(df, t)), Set(df, 0.5f));
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

// Computes x - y in double-float precision, assuming |x| > |y|
// Translated from common/df.h:134 dfsub_vf2_vf2_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SubDF(const D df, Vec2<D> x, Vec<D> y) {
  Vec<D> s = Sub(Get2<0>(x), y);
  return Create2(df, s, Add(Sub(Sub(Get2<0>(x), s), y), Get2<1>(x)));
}

// Zero out x when the sign bit of d is not set
// Translated from libm/sleefsimdsp.c:480 vsel_vi2_vf_vi2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<RebindToSigned<D>>) SignBitOrZero(const D df, Vec<D> d, Vec<RebindToSigned<D>> x) {
  RebindToSigned<D> di;
  
  return IfThenElseZero(RebindMask(di, SignBitMask(df, d)), x);
}

// Computes -x in double-float precision
// Translated from common/df.h:93 dfneg_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) NegDF(const D df, Vec2<D> x) {
  return Create2(df, Neg(Get2<0>(x)), Neg(Get2<1>(x)));
}

// atan2(x, y) in double-float precision
// Translated from libm/sleefsimdsp.c:1872 atan2kf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) ATan2DF(const D df, Vec2<D> y, Vec2<D> x) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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

// If d == inf, return m, if d == -inf, return -abs(m), otherwise return m
// Translated from libm/sleefsimdsp.c:1813 visinf2_vf_vf_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) IfInfThenElseZero(const D df, Vec<D> d, Vec<D> m) {
  RebindToUnsigned<D> du;
  
  return BitCast(df, IfThenElseZero(RebindMask(du, IsInf(d)), Or(SignBit(df, d), BitCast(du, m))));
}

// Calculate Atan2 without handling special cases
// Translated from libm/sleefsimdsp.c:1780 atan2kf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Atan2Helper(const D df, Vec<D> y, Vec<D> x) {
  RebindToSigned<D> di;
  
  Vec<D> s, t, u;
  Vec<RebindToSigned<D>> q;
  Mask<D> p;

  q = SignBitOrZero(df, x, Set(di, -2));
  x = Abs(x);

  q = IfThenElse(RebindMask(di, Lt(x, y)), Add(q, Set(di, 1)), q);
  p = Lt(x, y);
  s = IfThenElse(p, Neg(x), y);
  t = Max(x, y);

  s = Div(s, t);
  t = Mul(s, s);

  Vec<D> t2 = Mul(t, t), t4 = Mul(t2, t2);
  u = Estrin(t, t2, t4, Set(df, -0.333331018686294555664062f), Set(df, 0.199926957488059997558594f), Set(df, -0.142027363181114196777344f), Set(df, 0.106347933411598205566406f), Set(df, -0.0748900920152664184570312f), Set(df, 0.0425049886107444763183594f), Set(df, -0.0159569028764963150024414f), Set(df, 0.00282363896258175373077393f));

  t = MulAdd(s, Mul(t, u), s);
  t = MulAdd(ConvertTo(df, q), Set(df, (float)(Pi/2)), t);

  return t;
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

// Computes ln(x) in double-float precision (version 2)
// Translated from libm/sleefsimdsp.c:2526 logk2f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) LogFastDF(const D df, Vec2<D> d) {
  Vec2<D> x, x2, m, s;
  Vec<D> t;
  Vec<RebindToSigned<D>> e;

#if !(HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3)
  e = ILogB1(df, Mul(Get2<0>(d), Set(df, 1.0f/0.75f)));
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

// Computes x * y + z in double-float precision
// Translated from libm/sleefsimdsp.c:3461 dfmla_vf2_vf_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) MulAddDF(const D df, Vec<D> x, Vec2<D> y, Vec2<D> z) {
  return AddFastDF(df, z, MulDF(df, y, x));
}

// Computes 2nd-order polynomial in double-float precision
// Translated from libm/sleefsimdsp.c:3466 poly2df
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) Poly2DF(const D df, Vec<D> x, Vec<D> c1, Vec2<D> c0) {
 return MulAddDF(df, x, Create2(df, c1, Set(df, 0)), c0); }

// Computes 2nd-order polynomial in double-float precision
// Translated from libm/sleefsimdsp.c:3465 poly2df_b
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) Poly2DF(const D df, Vec<D> x, Vec2<D> c1, Vec2<D> c0) {
 return MulAddDF(df, x, c1, c0); }

// Computes 4th-order polynomial in double-float precision
// Translated from libm/sleefsimdsp.c:3467 poly4df
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) Poly4DF(const D df, Vec<D> x, Vec<D> c3, Vec2<D> c2, Vec2<D> c1, Vec2<D> c0) {
  return MulAddDF(df, Mul(x, x), Poly2DF(df, x, c3, c2), Poly2DF(df, x, c1, c0));
}

// Cast double into double-float precision
// Translated from common/df.h:34 vcast_vf2_d
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) CastDF(const D df, double d) {
  return Create2(df, Set(df, d), Set(df, d - (float)d));
}

// o0 ? d0 : (o1 ? d1 : (o2 ? d2 : d3))
// Translated from common/df.h:50 vsel_vf2_vo_vo_vo_d_d_d_d
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) IfThenElse4(const D df, Mask<D> o0, Mask<D> o1, Mask<D> o2, double d0, double d1, double d2, double d3) {
  return IfThenElse(df, o0, CastDF(df, d0), IfThenElse(df, o1, CastDF(df, d1), IfThenElse(df, o2, CastDF(df, d2), CastDF(df, d3))));
}

// o0 ? d0 : (o1 ? d1 : (o2 ? d2 : d3))
// Translated from arch/helperneon32.h:238 vsel_vf_vo_vo_vo_f_f_f_f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) IfThenElse4(const D df, Mask<D> o0, Mask<D> o1, Mask<D> o2, float d0, float d1, float d2, float d3) {
  return IfThenElse(o0, Set(df, d0), IfThenElse(o1, Set(df, d1), IfThenElse(o2, Set(df, d2), Set(df, d3))));
}

// o0 ? d0 : (o1 ? d1 : d2)
// Translated from arch/helperneon32.h:234 vsel_vf_vo_vo_f_f_f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) IfThenElse3(const D df, Mask<D> o0, Mask<D> o1, float d0, float d1, float d2) {
  return IfThenElse(o0, Set(df, d0), IfThenElse(o1, Set(df, d1), Set(df, d2)));
}

// Computes gamma(x) in quad-float precision
// Translated from libm/sleefsimdsp.c:3364 gammafk
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec4<D>) GammaQF(const D df, Vec<D> a) {
  RebindToSigned<D> di;
  
  Vec2<D> clc = Create2(df, Set(df, 0), Set(df, 0)), clln = Create2(df, Set(df, 1), Set(df, 0)), clld = Create2(df, Set(df, 1), Set(df, 0));
  Vec2<D> x, y, z;
  Vec<D> t, u;

  Mask<D> otiny = Lt(Abs(a), Set(df, 1e-30f)), oref = Lt(a, Set(df, 0.5));

  x = IfThenElse(df, otiny, Create2(df, Set(df, 0), Set(df, 0)),
			  IfThenElse(df, oref, AddDF(df, Set(df, 1), Neg(a)),
					      Create2(df, a, Set(df, 0))));

  Mask<D> o0 = And(Le(Set(df, 0.5), Get2<0>(x)), Le(Get2<0>(x), Set(df, 1.2)));
  Mask<D> o2 = Le(Set(df, 2.3), Get2<0>(x));
  
  y = NormalizeDF(df, MulDF(df, AddDF(df, x, Set(df, 1)), x));
  y = NormalizeDF(df, MulDF(df, AddDF(df, x, Set(df, 2)), y));

  Mask<D> o = And(o2, Le(Get2<0>(x), Set(df, 7)));
  clln = IfThenElse(df, o, y, clln);

  x = IfThenElse(df, o, AddDF(df, x, Set(df, 3)), x);
  t = IfThenElse(o2, Div(Set(df, 1.0), Get2<0>(x)), Get2<0>(NormalizeDF(df, AddDF(df, x, IfThenElse(o0, Set(df, -1), Set(df, -2))))));

  u = IfThenElse3(df, o2, o0, +0.000839498720672087279971000786, +0.9435157776e+0f, +0.1102489550e-3f);
  u = MulAdd(u, t, IfThenElse3(df, o2, o0, -5.17179090826059219329394422e-05, +0.8670063615e+0f, +0.8160019934e-4f));
  u = MulAdd(u, t, IfThenElse3(df, o2, o0, -0.000592166437353693882857342347, +0.4826702476e+0f, +0.1528468856e-3f));
  u = MulAdd(u, t, IfThenElse3(df, o2, o0, +6.97281375836585777403743539e-05, -0.8855129778e-1f, -0.2355068718e-3f));
  u = MulAdd(u, t, IfThenElse3(df, o2, o0, +0.000784039221720066627493314301, +0.1013825238e+0f, +0.4962242092e-3f));
  u = MulAdd(u, t, IfThenElse3(df, o2, o0, -0.000229472093621399176949318732, -0.1493408978e+0f, -0.1193488017e-2f));
  u = MulAdd(u, t, IfThenElse3(df, o2, o0, -0.002681327160493827160473958490, +0.1697509140e+0f, +0.2891599433e-2f));
  u = MulAdd(u, t, IfThenElse3(df, o2, o0, +0.003472222222222222222175164840, -0.2072454542e+0f, -0.7385451812e-2f));
  u = MulAdd(u, t, IfThenElse3(df, o2, o0, +0.083333333333333333335592087900, +0.2705872357e+0f, +0.2058077045e-1f));

  y = MulDF(df, AddDF(df, x, Set(df, -0.5)), LogFastDF(df, x));
  y = AddDF(df, y, NegDF(df, x));
  y = AddDF(df, y, CastDF(df, 0.91893853320467278056)); // 0.5*log(2*M_PI)

  z = AddDF(df, MulDF(df, u, t), IfThenElse(o0, Set(df, -0.400686534596170958447352690395e+0f), Set(df, -0.673523028297382446749257758235e-1f)));
  z = AddDF(df, MulDF(df, z, t), IfThenElse(o0, Set(df, +0.822466960142643054450325495997e+0f), Set(df, +0.322467033928981157743538726901e+0f)));
  z = AddDF(df, MulDF(df, z, t), IfThenElse(o0, Set(df, -0.577215665946766039837398973297e+0f), Set(df, +0.422784335087484338986941629852e+0f)));
  z = MulDF(df, z, t);

  clc = IfThenElse(df, o2, y, z);
  
  clld = IfThenElse(df, o2, AddDF(df, MulDF(df, u, t), Set(df, 1)), clld);
  
  y = clln;

  clc = IfThenElse(df, otiny, CastDF(df, 41.58883083359671856503), // log(2^60)
			    IfThenElse(df, oref, AddDF(df, CastDF(df, 1.1447298858494001639), NegDF(df, clc)), clc)); // log(M_PI)
  clln = IfThenElse(df, otiny, Create2(df, Set(df, 1), Set(df, 0)), IfThenElse(df, oref, clln, clld));

  if (!AllTrue(df, Not(oref))) {
    t = Sub(a, Mul(Set(df, INT64_C(1) << 12), ConvertTo(df, ConvertTo(di, Mul(a, Set(df, 1.0 / (INT64_C(1) << 12)))))));
    x = MulDF(df, clld, SinPiDF(df, t));
  }
  
  clld = IfThenElse(df, otiny, Create2(df, Mul(a, Set(df, (INT64_C(1) << 30)*(float)(INT64_C(1) << 30))), Set(df, 0)),
			     IfThenElse(df, oref, x, y));

  return Create4(df, Get2<0>(clc), Get2<1>(clc), Get2<0>(DivDF(df, clln, clld)), Get2<1>(DivDF(df, clln, clld)));
}

// True if d is an integer
// Translated from libm/sleefsimdsp.c:484 visint_vo_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Mask<D>) IsInt(const D df, Vec<D> y) {
 return Eq(Trunc(y), y); }

// Computes abs(x) in double-float precision
// Translated from common/df.h:97 dfabs_vf2_vf2
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) AbsDF(const D df, Vec2<D> x) {
  RebindToUnsigned<D> du;
  
  return Create2(df, Abs(Get2<0>(x)), BitCast(df, Xor(BitCast(du, Get2<1>(x)), And(BitCast(du, Get2<0>(x)), BitCast(du, Set(df, -0.0f))))));
}

// Take the next floating point number towards 0
// Translated from libm/sleefsimdsp.c:3158 vtoward0_vf_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Toward0(const D df, Vec<D> x) {
  RebindToSigned<D> di;
  
  Vec<D> t = BitCast(df, Sub(BitCast(di, x), Set(di, 1)));
  return IfThenElse(Eq(x, Set(df, 0)), Set(df, 0), t);
}

// Copy sign of y into x
// Translated from libm/sleefsimdsp.c:462 vcopysign_vf_vf_vf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) CopySign(const D df, Vec<D> x, Vec<D> y) {
  RebindToUnsigned<D> du;
  
  return BitCast(df, Xor(AndNot(BitCast(du, Set(df, -0.0f)), BitCast(du, x)), And(BitCast(du, Set(df, -0.0f)), BitCast(du, y))));
}

// Calculate exp(x) (presumably a faster, low-precision variant used in PowFaster)
// Translated from libm/sleefsimdsp.c:2337 expk3f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) ExpFast(const D df, Vec<D> d) {
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

  u = MulAdd(Mul(s, s), u, Add(s, Set(df, 1.0f)));
  u = LoadExp2(df, u, q);

  u = BitCast(df, IfThenZeroElse(RebindMask(du, Lt(d, Set(df, -104))), BitCast(du, u)));
  
  return u;
}

// Calculate log(x) (presumably a faster, low-precision variant used in PowFaster)
// Translated from libm/sleefsimdsp.c:2234 logk3f
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
#else
  x = MulAdd(x, t, Mul(Set(df, 0.693147180559945286226764f), e));
#endif

  return x;
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
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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

// Integer log of x (helper, not top-level)
// Translated from common/commonfuncs.h:290 vilogbk_vi_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<RebindToSigned<D>>) ILogB1(const D df, Vec<D> d) {
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
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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

// Computes x * y in double-double precision, returning result as double
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
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) PayneHanekReductionHelper(const D df, Vec<D> x) {
  RebindToSigned<D> di;
  
Vec<D> y = Round(Mul(x, Set(df, 4)));
  Vec<RebindToSigned<D>> vi = ConvertTo(di, Trunc(Sub(y, Mul(Round(x), Set(df, 4)))));
  return Create2(df, Sub(x, Mul(y, Set(df, 0.25))), BitCast(df, vi));

}

// Calculate Payne Hanek reduction. This appears to return ((2*x/pi) - round(2*x/pi)) * pi / 2 and the integer quadrant of x in range -2 to 2 (0 is [-pi/4, pi/4], 2/-2 are from [3pi/4, 5pi/4] with the sign flip a little after pi).
// Translated from libm/sleefsimddp.c:348 rempi
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec3<D>) PayneHanekReduction(const D df, Vec<D> a) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Vec2<D> x, y;
  Vec<RebindToSigned<D>> ex = ILogB2(df, a);
#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
  ex = AndNot(ShiftRight<63>(ex), ex);
  ex = And(ex, Set(di, 1023));
#endif
  ex = Sub(ex, Set(di, 55));
  Vec<RebindToSigned<D>> q = IfThenElseZero(RebindMask(di, Gt(ex, Set(di, 700-55))), Set(di, -64));
  a = LoadExp3(df, a, q);
  ex = AndNot(ShiftRight<63>(ex), ex);
  ex = ShiftLeft<2>(ex);
  x = MulDD(df, a, GatherIndex(df, PayneHanekReductionTable_double, ex));
  Vec2<D> di_ = PayneHanekReductionHelper(df, Get2<0>(x));
  q = BitCast(di, Get2<1>(di_));
  x = Set2<0>(x, Get2<0>(di_));
  x = NormalizeDD(df, x);
  y = MulDD(df, a, GatherIndex(df, PayneHanekReductionTable_double+1, ex));
  x = AddDD(df, x, y);
  di_ = PayneHanekReductionHelper(df, Get2<0>(x));
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

// Computes -x in double-double precision
// Translated from common/dd.h:97 ddneg_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) NegDD(const D df, Vec2<D> x) {
  return Create2(df, Neg(Get2<0>(x)), Neg(Get2<1>(x)));
}

// Computes sqrt(x) in double-double precision
// Translated from common/dd.h:317 ddsqrt_vd2_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SqrtDD(const D df, Vec<D> d) {
  Vec<D> t = Sqrt(d);
  return ScaleDD(df, MulDD(df, AddDD(df, d, MulDD(df, t, t)), RecDD(df, t)), Set(df, 0.5));
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

// Computes x - y in double-double precision, assuming |x| > |y|
// Translated from common/dd.h:135 ddsub_vd2_vd2_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SubDD(const D df, Vec2<D> x, Vec<D> y) {
  Vec<D> s = Sub(Get2<0>(x), y);
  return Create2(df, s, Add(Sub(Sub(Get2<0>(x), s), y), Get2<1>(x)));
}

// d0 < d1 ? x : y
// Translated from libm/sleefsimddp.c:331 vsel_vi_vd_vd_vi_vi
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<RebindToSigned<D>>) IfLtThenElseZero(const D df, Vec<D> d0, Vec<D> d1, Vec<RebindToSigned<D>> x, Vec<RebindToSigned<D>> y) {
  RebindToSigned<D> di;
  
 return IfThenElse(RebindMask(di, Lt(d0, d1)), x, y); }

// d0 < 0 ? x : y
// Translated from libm/sleefsimddp.c:334 vsel_vi_vd_vi
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<RebindToSigned<D>>) IfNegThenElseZero(const D df, Vec<D> d, Vec<RebindToSigned<D>> x) {
  RebindToSigned<D> di;
  
 return IfThenElseZero(RebindMask(di, SignBitMask(df, d)), x); }

// atan2(x, y) in double-double precision
// Translated from libm/sleefsimddp.c:1835 atan2k_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) ATan2DD(const D df, Vec2<D> y, Vec2<D> x) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Vec<D> u;
  Vec2<D> s, t;
  Vec<RebindToSigned<D>> q;
  Mask<D> p;

  q = IfNegThenElseZero(df, Get2<0>(x), Set(di, -2));
  p = Lt(Get2<0>(x), Set(df, 0));
  Vec<RebindToUnsigned<D>> b = IfThenElseZero(RebindMask(du, p), BitCast(du, Set(df, -0.0)));
  x = Set2<0>(x, BitCast(df, Xor(b, BitCast(du, Get2<0>(x)))));
  x = Set2<1>(x, BitCast(df, Xor(b, BitCast(du, Get2<1>(x)))));

  q = IfLtThenElseZero(df, Get2<0>(x), Get2<0>(y), Add(q, Set(di, 1)), q);
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

// If d == inf, return m, if d == -inf, return -abs(m), otherwise return m
// Translated from libm/sleefsimddp.c:1886 visinf2_vd_vd_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) IfInfThenElseZero(const D df, Vec<D> d, Vec<D> m) {
  RebindToUnsigned<D> du;
  
  return BitCast(df, IfThenElseZero(RebindMask(du, IsInf(d)), Or(And(BitCast(du, d), BitCast(du, Set(df, -0.0))), BitCast(du, m))));
}

// Calculate Atan2 without handling special cases
// Translated from libm/sleefsimddp.c:1791 atan2k
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Atan2Helper(const D df, Vec<D> y, Vec<D> x) {
  RebindToSigned<D> di;
  
  Vec<D> s, t, u;
  Vec<RebindToSigned<D>> q;
  Mask<D> p;

  q = IfNegThenElseZero(df, x, Set(di, -2));
  x = Abs(x);

  q = IfLtThenElseZero(df, x, y, Add(q, Set(di, 1)), q);
  p = Lt(x, y);
  s = IfThenElse(p, Neg(x), y);
  t = Max(x, y);

  s = Div(s, t);
  t = Mul(s, s);

  Vec<D> t2 = Mul(t, t), t4 = Mul(t2, t2), t8 = Mul(t4, t4), t16 = Mul(t8, t8);
  u = Estrin(t, t2, t4, t8, t16, Set(df, -0.333333333333311110369124), Set(df, 0.199999999996591265594148), Set(df, -0.14285714266771329383765), Set(df, 0.111111105648261418443745), Set(df, -0.090908995008245008229153), Set(df, 0.0769219538311769618355029), Set(df, -0.0666573579361080525984562), Set(df, 0.0587666392926673580854313), Set(df, -0.0523674852303482457616113), Set(df, 0.0466667150077840625632675), Set(df, -0.0407629191276836500001934), Set(df, 0.0337852580001353069993897), Set(df, -0.0254517624932312641616861), Set(df, 0.016599329773529201970117), Set(df, -0.00889896195887655491740809), Set(df, 0.00370026744188713119232403), Set(df, -0.00110611831486672482563471), Set(df, 0.000209850076645816976906797), Set(df, -1.88796008463073496563746e-05));
  
  t = MulAdd(s, Mul(t, u), s);
  t = MulAdd(ConvertTo(df, q), Set(df, Pi/2), t);

  return t;
}

// Computes x^2 in double-double precision, returning result as double
// Translated from common/dd.h:218 ddsqu_vd_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) SquareDD_double(const D df, Vec2<D> x) {
#if HWY_SLEEF_HAS_FMA
  return MulAdd(Get2<0>(x), Get2<0>(x), Add(Mul(Get2<0>(x), Get2<1>(x)), Mul(Get2<0>(x), Get2<1>(x))));
#else
  Vec<D> xh = LowerPrecision(df, Get2<0>(x)), xl = Sub(Get2<0>(x), xh);

  return Add5(df, Mul(xh, Get2<1>(x)), Mul(xh, Get2<1>(x)), Mul(xl, xl), Add(Mul(xh, xl), Mul(xh, xl)), Mul(xh, xh));
#endif
}

// o ? (x1, y1) : (x0, y0)
// Translated from common/dd.h:54 vsel_vd2_vo_d_d_d_d
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) IfThenElse(const D df, Mask<D> o, double x1, double y1, double x0, double y0) {
  return Create2(df, IfThenElse(o, Set(df, x1), Set(df, x0)), IfThenElse(o, Set(df, y1), Set(df, y0)));
}

// Calculate SinPi without handling special cases in double-double precision
// Translated from libm/sleefsimddp.c:1416 sinpik
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SinPiDD(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Mask<D> o;
  Vec<D> u, s, t;
  Vec2<D> x, s2;

  u = Mul(d, Set(df, 4.0));
  Vec<RebindToSigned<D>> q = ConvertTo(di, Trunc(u));
  q = And(Add(q, Xor(BitCast(di, ShiftRight<63>(BitCast(du, q))), Set(di, 1))), Set(di, ~1));
  o = RebindMask(df, Eq(And(q, Set(di, 2)), Set(di, 2)));

  s = Sub(u, ConvertTo(df, q));
  t = s;
  s = Mul(s, s);
  s2 = MulDD(df, t, t);

  

  u = IfThenElse(o, Set(df, 9.94480387626843774090208e-16), Set(df, -2.02461120785182399295868e-14));
  u = MulAdd(u, s, IfThenElse(o, Set(df, -3.89796226062932799164047e-13), Set(df, 6.948218305801794613277840e-12)));
  u = MulAdd(u, s, IfThenElse(o, Set(df, 1.150115825399960352669010e-10), Set(df, -1.75724749952853179952664e-09)));
  u = MulAdd(u, s, IfThenElse(o, Set(df, -2.46113695010446974953590e-08), Set(df, 3.133616889668683928784220e-07)));
  u = MulAdd(u, s, IfThenElse(o, Set(df, 3.590860448590527540050620e-06), Set(df, -3.65762041821615519203610e-05)));
  u = MulAdd(u, s, IfThenElse(o, Set(df, -0.000325991886927389905997954), Set(df, 0.0024903945701927185027435600)));
  x = AddDD(df, Mul(u, s),
			IfThenElse(df, o, 0.0158543442438155018914259, -1.04693272280631521908845e-18,
					    -0.0807455121882807852484731, 3.61852475067037104849987e-18));
  x = AddDD(df, MulDD(df, s2, x),
			 IfThenElse(df, o, -0.308425137534042437259529, -1.95698492133633550338345e-17,
					     0.785398163397448278999491, 3.06287113727155002607105e-17));

  x = MulDD(df, x, IfThenElse(df, o, s2, Create2(df, t, Set(df, 0))));
  x = IfThenElse(df, o, AddDD(df, x, Set(df, 1)), x);

  o = RebindMask(df, Eq(And(q, Set(di, 4)), Set(di, 4)));
  x = Set2<0>(x, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<0>(x)))));
  x = Set2<1>(x, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<1>(x)))));

  return x;
}

// Calculate CosPi without handling special cases in double-double precision
// Translated from libm/sleefsimddp.c:1467 cospik
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) CosPiDD(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Mask<D> o;
  Vec<D> u, s, t;
  Vec2<D> x, s2;

  u = Mul(d, Set(df, 4.0));
  Vec<RebindToSigned<D>> q = ConvertTo(di, Trunc(u));
  q = And(Add(q, Xor(BitCast(di, ShiftRight<63>(BitCast(du, q))), Set(di, 1))), Set(di, ~1));
  o = RebindMask(df, Eq(And(q, Set(di, 2)), Set(di, 0)));

  s = Sub(u, ConvertTo(df, q));
  t = s;
  s = Mul(s, s);
  s2 = MulDD(df, t, t);
  
  

  u = IfThenElse(o, Set(df, 9.94480387626843774090208e-16), Set(df, -2.02461120785182399295868e-14));
  u = MulAdd(u, s, IfThenElse(o, Set(df, -3.89796226062932799164047e-13), Set(df, 6.948218305801794613277840e-12)));
  u = MulAdd(u, s, IfThenElse(o, Set(df, 1.150115825399960352669010e-10), Set(df, -1.75724749952853179952664e-09)));
  u = MulAdd(u, s, IfThenElse(o, Set(df, -2.46113695010446974953590e-08), Set(df, 3.133616889668683928784220e-07)));
  u = MulAdd(u, s, IfThenElse(o, Set(df, 3.590860448590527540050620e-06), Set(df, -3.65762041821615519203610e-05)));
  u = MulAdd(u, s, IfThenElse(o, Set(df, -0.000325991886927389905997954), Set(df, 0.0024903945701927185027435600)));
  x = AddDD(df, Mul(u, s),
			IfThenElse(df, o, 0.0158543442438155018914259, -1.04693272280631521908845e-18,
					    -0.0807455121882807852484731, 3.61852475067037104849987e-18));
  x = AddDD(df, MulDD(df, s2, x),
			 IfThenElse(df, o, -0.308425137534042437259529, -1.95698492133633550338345e-17,
					     0.785398163397448278999491, 3.06287113727155002607105e-17));

  x = MulDD(df, x, IfThenElse(df, o, s2, Create2(df, t, Set(df, 0))));
  x = IfThenElse(df, o, AddDD(df, x, Set(df, 1)), x);

  o = RebindMask(df, Eq(And(Add(q, Set(di, 2)), Set(di, 4)), Set(di, 4)));
  x = Set2<0>(x, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<0>(x)))));
  x = Set2<1>(x, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<1>(x)))));

  return x;
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

// Computes e^x - 1 faster with lower precision
// Translated from libm/sleefsimddp.c:2195 expm1k
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Expm1Fast(const D df, Vec<D> d) {
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

// Computes ln(x) in double-double precision (version 2)
// Translated from libm/sleefsimddp.c:2508 logk2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) LogFastDD(const D df, Vec2<D> d) {
  Vec2<D> x, x2, m, s;
  Vec<D> t;
  Vec<RebindToSigned<D>> e;
  
  e = ILogB1(df, Mul(Get2<0>(d), Set(df, 1.0/0.75)));

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

// Computes x * y + z in double-float precision
// Translated from libm/sleefsimddp.c:3460 ddmla_vd2_vd_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) MulAddDD(const D df, Vec<D> x, Vec2<D> y, Vec2<D> z) {
  return AddFastDD(df, z, MulDD(df, y, x));
}

// Computes 2nd-order polynomial in double-float precision
// Translated from libm/sleefsimddp.c:3465 poly2dd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) Poly2DD(const D df, Vec<D> x, Vec<D> c1, Vec2<D> c0) {
 return MulAddDD(df, x, Create2(df, c1, Set(df, 0)), c0); }

// Computes 2nd-order polynomial in double-float precision
// Translated from libm/sleefsimddp.c:3464 poly2dd_b
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) Poly2DD(const D df, Vec<D> x, Vec2<D> c1, Vec2<D> c0) {
 return MulAddDD(df, x, c1, c0); }

// Computes 4th-order polynomial in double-double precision
// Translated from libm/sleefsimddp.c:3466 poly4dd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) Poly4DD(const D df, Vec<D> x, Vec<D> c3, Vec2<D> c2, Vec2<D> c1, Vec2<D> c0) {
  return MulAddDD(df, Mul(x, x), Poly2DD(df, x, c3, c2), Poly2DD(df, x, c1, c0));
}

// Computes gamma(x) in quad-double precision
// Translated from libm/sleefsimddp.c:3347 gammak
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec4<D>) GammaQD(const D df, Vec<D> a) {
  RebindToSigned<D> di;
  
  Vec2<D> clc = Create2(df, Set(df, 0), Set(df, 0)), clln = Create2(df, Set(df, 1), Set(df, 0)), clld = Create2(df, Set(df, 1), Set(df, 0));
  Vec2<D> x, y, z;
  Vec<D> t, u;

  Mask<D> otiny = Lt(Abs(a), Set(df, 1e-306)), oref = Lt(a, Set(df, 0.5));

  x = IfThenElse(df, otiny, Create2(df, Set(df, 0), Set(df, 0)),
			  IfThenElse(df, oref, AddDD(df, Set(df, 1), Neg(a)),
					      Create2(df, a, Set(df, 0))));

  Mask<D> o0 = And(Le(Set(df, 0.5), Get2<0>(x)), Le(Get2<0>(x), Set(df, 1.1)));
  Mask<D> o2 = Le(Set(df, 2.3), Get2<0>(x));
  
  y = NormalizeDD(df, MulDD(df, AddDD(df, x, Set(df, 1)), x));
  y = NormalizeDD(df, MulDD(df, AddDD(df, x, Set(df, 2)), y));
  y = NormalizeDD(df, MulDD(df, AddDD(df, x, Set(df, 3)), y));
  y = NormalizeDD(df, MulDD(df, AddDD(df, x, Set(df, 4)), y));

  Mask<D> o = And(o2, Le(Get2<0>(x), Set(df, 7)));
  clln = IfThenElse(df, o, y, clln);

  x = IfThenElse(df, o, AddDD(df, x, Set(df, 5)), x);
  
  t = IfThenElse(o2, Div(Set(df, 1.0), Get2<0>(x)), Get2<0>(NormalizeDD(df, AddDD(df, x, IfThenElse(o0, Set(df, -1), Set(df, -2))))));

  u = IfThenElse(o2, Set(df, -156.801412704022726379848862), IfThenElse(o0, Set(df, +0.2947916772827614196e+2), Set(df, +0.7074816000864609279e-7)));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, +1.120804464289911606838558160000), IfThenElse(o0, Set(df, +0.1281459691827820109e+3), Set(df, +0.4009244333008730443e-6))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, +13.39798545514258921833306020000), IfThenElse(o0, Set(df, +0.2617544025784515043e+3), Set(df, +0.1040114641628246946e-5))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, -0.116546276599463200848033357000), IfThenElse(o0, Set(df, +0.3287022855685790432e+3), Set(df, +0.1508349150733329167e-5))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, -1.391801093265337481495562410000), IfThenElse(o0, Set(df, +0.2818145867730348186e+3), Set(df, +0.1288143074933901020e-5))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, +0.015056113040026424412918973400), IfThenElse(o0, Set(df, +0.1728670414673559605e+3), Set(df, +0.4744167749884993937e-6))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, +0.179540117061234856098844714000), IfThenElse(o0, Set(df, +0.7748735764030416817e+2), Set(df, -0.6554816306542489902e-7))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, -0.002481743600264997730942489280), IfThenElse(o0, Set(df, +0.2512856643080930752e+2), Set(df, -0.3189252471452599844e-6))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, -0.029527880945699120504851034100), IfThenElse(o0, Set(df, +0.5766792106140076868e+1), Set(df, +0.1358883821470355377e-6))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, +0.000540164767892604515196325186), IfThenElse(o0, Set(df, +0.7270275473996180571e+0), Set(df, -0.4343931277157336040e-6))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, +0.006403362833808069794787256200), IfThenElse(o0, Set(df, +0.8396709124579147809e-1), Set(df, +0.9724785897406779555e-6))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, -0.000162516262783915816896611252), IfThenElse(o0, Set(df, -0.8211558669746804595e-1), Set(df, -0.2036886057225966011e-5))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, -0.001914438498565477526465972390), IfThenElse(o0, Set(df, +0.6828831828341884458e-1), Set(df, +0.4373363141819725815e-5))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, +7.20489541602001055898311517e-05), IfThenElse(o0, Set(df, -0.7712481339961671511e-1), Set(df, -0.9439951268304008677e-5))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, +0.000839498720672087279971000786), IfThenElse(o0, Set(df, +0.8337492023017314957e-1), Set(df, +0.2050727030376389804e-4))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, -5.17179090826059219329394422e-05), IfThenElse(o0, Set(df, -0.9094964931456242518e-1), Set(df, -0.4492620183431184018e-4))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, -0.000592166437353693882857342347), IfThenElse(o0, Set(df, +0.1000996313575929358e+0), Set(df, +0.9945751236071875931e-4))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, +6.97281375836585777403743539e-05), IfThenElse(o0, Set(df, -0.1113342861544207724e+0), Set(df, -0.2231547599034983196e-3))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, +0.000784039221720066627493314301), IfThenElse(o0, Set(df, +0.1255096673213020875e+0), Set(df, +0.5096695247101967622e-3))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, -0.000229472093621399176949318732), IfThenElse(o0, Set(df, -0.1440498967843054368e+0), Set(df, -0.1192753911667886971e-2))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, -0.002681327160493827160473958490), IfThenElse(o0, Set(df, +0.1695571770041949811e+0), Set(df, +0.2890510330742210310e-2))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, +0.003472222222222222222175164840), IfThenElse(o0, Set(df, -0.2073855510284092762e+0), Set(df, -0.7385551028674461858e-2))));
  u = MulAdd(u, t, IfThenElse(o2, Set(df, +0.083333333333333333335592087900), IfThenElse(o0, Set(df, +0.2705808084277815939e+0), Set(df, +0.2058080842778455335e-1))));

  y = MulDD(df, AddDD(df, x, Set(df, -0.5)), LogFastDD(df, x));
  y = AddDD(df, y, NegDD(df, x));
  y = AddDD(df, y, Create2(df, Set(df, 0.91893853320467278056), Set(df, -3.8782941580672414498e-17))); // 0.5*log(2*M_PI)

  z = AddDD(df, MulDD(df, u, t), IfThenElse(o0, Set(df, -0.4006856343865314862e+0), Set(df, -0.6735230105319810201e-1)));
  z = AddDD(df, MulDD(df, z, t), IfThenElse(o0, Set(df, +0.8224670334241132030e+0), Set(df, +0.3224670334241132030e+0)));
  z = AddDD(df, MulDD(df, z, t), IfThenElse(o0, Set(df, -0.5772156649015328655e+0), Set(df, +0.4227843350984671345e+0)));
  z = MulDD(df, z, t);

  clc = IfThenElse(df, o2, y, z);
  
  clld = IfThenElse(df, o2, AddDD(df, MulDD(df, u, t), Set(df, 1)), clld);
  
  y = clln;

  clc = IfThenElse(df, otiny, Create2(df, Set(df, 83.1776616671934334590333), Set(df, 3.67103459631568507221878e-15)), // log(2^120)
			    IfThenElse(df, oref, AddDD(df, Create2(df, Set(df, 1.1447298858494001639), Set(df, 1.026595116270782638e-17)), NegDD(df, clc)), clc)); // log(M_PI)
  clln = IfThenElse(df, otiny, Create2(df, Set(df, 1), Set(df, 0)), IfThenElse(df, oref, clln, clld));

  if (!AllTrue(df, Not(oref))) {
    t = Sub(a, Mul(Set(df, INT64_C(1) << 28), ConvertTo(df, ConvertTo(di, Trunc(Mul(a, Set(df, 1.0 / (INT64_C(1) << 28))))))));
    x = MulDD(df, clld, SinPiDD(df, t));
  }
  
  clld = IfThenElse(df, otiny, Create2(df, Mul(a, Set(df, (INT64_C(1) << 60)*(double)(INT64_C(1) << 60))), Set(df, 0)),
			     IfThenElse(df, oref, x, y));

  return Create4(df, Get2<0>(clc), Get2<1>(clc), Get2<0>(DivDD(df, clln, clld)), Get2<1>(DivDD(df, clln, clld)));
}

// Computes abs(x) in double-double precision
// Translated from common/dd.h:101 ddabs_vd2_vd2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) AbsDD(const D df, Vec2<D> x) {
  RebindToUnsigned<D> du;
  
  return Create2(df, Abs(Get2<0>(x)), BitCast(df, Xor(BitCast(du, Get2<1>(x)), And(BitCast(du, Get2<0>(x)), BitCast(du, Set(df, -0.0))))));
}

// Take the next floating point number towards 0
// Translated from common/commonfuncs.h:208 vtoward0_vd_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Toward0(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
 // returns nextafter(x, 0)
  Vec<D> t = BitCast(df, Add(BitCast(du, x), Set(du, -1)));
  return IfThenElse(Eq(x, Set(df, 0)), Set(df, 0), t);
}

// Sets the exponent of 'x' to 2^e
// Translated from common/commonfuncs.h:337 vldexp_vd_vd_vi
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) LoadExp(const D df, Vec<D> x, Vec<RebindToSigned<D>> q) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Vec<RebindToSigned<D>> m = ShiftRight<63>(q);
  m = ShiftLeft<7>(Sub(ShiftRight<9>(Add(m, q)), m));
  q = Sub(q, ShiftLeft<2>(m));
  m = Add(Set(di, 0x3ff), m);
  m = IfThenZeroElse(RebindMask(di, Gt(Set(di, 0), m)), m);
  m = IfThenElse(RebindMask(di, Gt(m, Set(di, 0x7ff))), Set(di, 0x7ff), m);
  Vec<RebindToUnsigned<D>> r = ShiftLeft<32>(BitCast(du, ShiftLeft<20>(m)));
  Vec<D> y = BitCast(df, r);
  return Mul(Mul(Mul(Mul(Mul(x, y), y), y), y), Pow2I(df, q));
}

// Copy sign of y into x
// Translated from common/commonfuncs.h:228 vcopysign_vd_vd_vd
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) CopySign(const D df, Vec<D> x, Vec<D> y) {
  RebindToUnsigned<D> du;
  
  return BitCast(df, Xor(AndNot(BitCast(du, Set(df, -0.0)), BitCast(du, x)), And(BitCast(du, Set(df, -0.0)), BitCast(du, y))));
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
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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

// Computes sqrt(x) with 0.5001 ULP accuracy
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

  y = BitCast(df, Sub(Set(di, 0x5f3759df), BitCast(di, ShiftRight<1>(BitCast(du, d)))));

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

  Vec<D> x = BitCast(df, Sub(Set(di, 0x5f375a86), BitCast(di, ShiftRight<1>(BitCast(du, Add(d, Set(df, 1e-45f)))))));

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
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Vec<D> e = BitCast(df, Add(Set(di, 0x20000000), And(Set(di, 0x7f000000), BitCast(di, ShiftRight<1>(BitCast(du, d))))));
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
  e = Add(ILogB1(df, Abs(d)), Set(di, 1));
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
  e = Add(ILogB1(df, Abs(d)), Set(di, 1));
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

// Computes sqrt(x^2 + y^2) with 0.5001 ULP accuracy
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
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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

  u = BitCast(df, Xor(IfThenElseZero(RebindMask(du, Eq(And(q, Set(di, 1)), Set(di, 1))), BitCast(du, Set(df, -0.0))), BitCast(du, u)));

  u = IfThenElse(Eq(d, Set(df, -0.0)), d, u);

  return u; 
}

// Computes cos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1067 xcosf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Cos(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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

  u = BitCast(df, Xor(IfThenElseZero(RebindMask(du, Eq(And(q, Set(di, 2)), Set(di, 0))), BitCast(du, Set(df, -0.0))), BitCast(du, u)));
  
  return u; 
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
  
  return u; 
}

// Computes sin(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:630 xsinf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) SinFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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

  d = BitCast(df, Xor(IfThenElseZero(RebindMask(du, Eq(And(q, Set(di, 1)), Set(di, 1))), BitCast(du, Set(df, -0.0f))), BitCast(du, d)));

  u = Set(df, 2.6083159809786593541503e-06f);
  u = MulAdd(u, s, Set(df, -0.0001981069071916863322258f));
  u = MulAdd(u, s, Set(df, 0.00833307858556509017944336f));
  u = MulAdd(u, s, Set(df, -0.166666597127914428710938f));

  u = Add(Mul(s, Mul(u, d)), d);

  u = IfThenElse(Eq(r, Set(df, -0.0)), r, u);

  return u; 
}

// Computes cos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:736 xcosf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) CosFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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

  d = BitCast(df, Xor(IfThenElseZero(RebindMask(du, Eq(And(q, Set(di, 2)), Set(di, 0))), BitCast(du, Set(df, -0.0f))), BitCast(du, d)));

  u = Set(df, 2.6083159809786593541503e-06f);
  u = MulAdd(u, s, Set(df, -0.0001981069071916863322258f));
  u = MulAdd(u, s, Set(df, 0.00833307858556509017944336f));
  u = MulAdd(u, s, Set(df, -0.166666597127914428710938f));

  u = Add(Mul(s, Mul(u, d)), d);

  return u; 
}

// Computes tan(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:845 xtanf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) TanFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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

  return u; 
}

// Computes sin(x) and cos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1360 XSINCOSF_U1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SinCos(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<RebindToSigned<D>> q;
  Mask<D> o;
  Vec<D> u, v, rx, ry;
  Vec2<D> r, s, t, x;

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
    q = IfThenElse(RebindMask(di, g), q, BitCast(di, Get3<2>(dfi)));
    s = IfThenElse(df, g, s, t);
  }

  t = s;

  s = Set2<0>(s, SquareDF_float(df, s));

  u = Set(df, -0.000195169282960705459117889f);
  u = MulAdd(u, Get2<0>(s), Set(df, 0.00833215750753879547119141f));
  u = MulAdd(u, Get2<0>(s), Set(df, -0.166666537523269653320312f));

  u = Mul(u, Mul(Get2<0>(s), Get2<0>(t)));

  x = AddFastDF(df, t, u);
  rx = Add(Get2<0>(x), Get2<1>(x));

  rx = IfThenElse(Eq(d, Set(df, -0.0)), Set(df, -0.0f), rx);

  u = Set(df, -2.71811842367242206819355e-07f);
  u = MulAdd(u, Get2<0>(s), Set(df, 2.47990446951007470488548e-05f));
  u = MulAdd(u, Get2<0>(s), Set(df, -0.00138888787478208541870117f));
  u = MulAdd(u, Get2<0>(s), Set(df, 0.0416666641831398010253906f));
  u = MulAdd(u, Get2<0>(s), Set(df, -0.5));

  x = AddFastDF(df, Set(df, 1), MulDF(df, Get2<0>(s), u));
  ry = Add(Get2<0>(x), Get2<1>(x));

  o = RebindMask(df, Eq(And(q, Set(di, 1)), Set(di, 0)));
  r = Create2(df, IfThenElse(o, rx, ry), IfThenElse(o, ry, rx));

  o = RebindMask(df, Eq(And(q, Set(di, 2)), Set(di, 2)));
  r = Set2<0>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<0>(r)))));

  o = RebindMask(df, Eq(And(Add(q, Set(di, 1)), Set(di, 2)), Set(di, 2)));
  r = Set2<1>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<1>(r)))));

  return r; 
}

// Computes sin(x) and cos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1233 XSINCOSF
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SinCosFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Vec<RebindToSigned<D>> q;
  Mask<D> o;
  Vec<D> u, s, t, rx, ry;
  Vec2<D> r;

  q = NearestInt(Mul(d, Set(df, (float)TwoOverPi)));
  u = ConvertTo(df, q);
  s = MulAdd(u, Set(df, -PiA2f*0.5f), d);
  s = MulAdd(u, Set(df, -PiB2f*0.5f), s);
  s = MulAdd(u, Set(df, -PiC2f*0.5f), s);
  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2f));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    Vec<RebindToSigned<D>> q2 = NearestInt(Mul(d, Set(df, (float)TwoOverPi)));
    u = ConvertTo(df, q2);
    t = MulAdd(u, Set(df, -PiAf*0.5f), d);
    t = MulAdd(u, Set(df, -PiBf*0.5f), t);
    t = MulAdd(u, Set(df, -PiCf*0.5f), t);
    t = MulAdd(u, Set(df, -PiDf*0.5f), t);

    q = IfThenElse(RebindMask(di, g), q, q2);
    s = IfThenElse(g, s, t);
    g = Lt(Abs(d), Set(df, TrigRangeMaxf));

    if (!HWY_LIKELY(AllTrue(df, g))) {
      Vec3<D> dfi = PayneHanekReduction(df, d);
      t = Add(Get2<0>(Create2(df, Get3<0>(dfi), Get3<1>(dfi))), Get2<1>(Create2(df, Get3<0>(dfi), Get3<1>(dfi))));
      t = BitCast(df, IfThenElse(RebindMask(du, Or(IsInf(d), IsNaN(d))), Set(du, -1), BitCast(du, t)));

      q = IfThenElse(RebindMask(di, g), q, BitCast(di, Get3<2>(dfi)));
      s = IfThenElse(g, s, t);
    }
  }

  t = s;

  s = Mul(s, s);

  u = Set(df, -0.000195169282960705459117889f);
  u = MulAdd(u, s, Set(df, 0.00833215750753879547119141f));
  u = MulAdd(u, s, Set(df, -0.166666537523269653320312f));

  rx = MulAdd(Mul(u, s), t, t);
  rx = IfThenElse(Eq(d, Set(df, -0.0)), Set(df, -0.0f), rx);

  u = Set(df, -2.71811842367242206819355e-07f);
  u = MulAdd(u, s, Set(df, 2.47990446951007470488548e-05f));
  u = MulAdd(u, s, Set(df, -0.00138888787478208541870117f));
  u = MulAdd(u, s, Set(df, 0.0416666641831398010253906f));
  u = MulAdd(u, s, Set(df, -0.5));

  ry = MulAdd(s, u, Set(df, 1));

  o = RebindMask(df, Eq(And(q, Set(di, 1)), Set(di, 0)));
  r = Create2(df, IfThenElse(o, rx, ry), IfThenElse(o, ry, rx));

  o = RebindMask(df, Eq(And(q, Set(di, 2)), Set(di, 2)));
  r = Set2<0>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<0>(r)))));

  o = RebindMask(df, Eq(And(Add(q, Set(di, 1)), Set(di, 2)), Set(di, 2)));
  r = Set2<1>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<1>(r)))));

  return r; 
}

// Computes sin(x*pi) and cos(x*pi) with max(0.506 ULP, FLT_MIN) accuracy (in practice staying below 2 ULP even for subnormals)
// Translated from libm/sleefsimdsp.c:1477 XSINCOSPIF_U05
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SinCosPi(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Mask<D> o;
  Vec<D> u, s, t, rx, ry;
  Vec2<D> r, x, s2;

  u = Mul(d, Set(df, 4));
  Vec<RebindToSigned<D>> q = ConvertTo(di, u);
  q = And(Add(q, Xor(BitCast(di, ShiftRight<31>(BitCast(du, q))), Set(di, 1))), Set(di, ~1));
  s = Sub(u, ConvertTo(df, q));

  t = s;
  s = Mul(s, s);
  s2 = MulDF(df, t, t);
  
  

  u = Set(df, +0.3093842054e-6);
  u = MulAdd(u, s, Set(df, -0.3657307388e-4));
  u = MulAdd(u, s, Set(df, +0.2490393585e-2));
  x = AddDF(df, Mul(u, s), Create2(df, Set(df, -0.080745510756969451904), Set(df, -1.3373665339076936258e-09)));
  x = AddDF(df, MulDF(df, s2, x), Create2(df, Set(df, 0.78539818525314331055), Set(df, -2.1857338617566484855e-08)));

  x = MulDF(df, x, t);
  rx = Add(Get2<0>(x), Get2<1>(x));

  rx = IfThenElse(Eq(d, Set(df, -0.0)), Set(df, -0.0f), rx);
  
  
  
  u = Set(df, -0.2430611801e-7);
  u = MulAdd(u, s, Set(df, +0.3590577080e-5));
  u = MulAdd(u, s, Set(df, -0.3259917721e-3));
  x = AddDF(df, Mul(u, s), Create2(df, Set(df, 0.015854343771934509277), Set(df, 4.4940051354032242811e-10)));
  x = AddDF(df, MulDF(df, s2, x), Create2(df, Set(df, -0.30842512845993041992), Set(df, -9.0728339030733922277e-09)));

  x = AddDF(df, MulDF(df, x, s2), Set(df, 1));
  ry = Add(Get2<0>(x), Get2<1>(x));

  

  o = RebindMask(df, Eq(And(q, Set(di, 2)), Set(di, 0)));
  r = Create2(df, IfThenElse(o, rx, ry), IfThenElse(o, ry, rx));

  o = RebindMask(df, Eq(And(q, Set(di, 4)), Set(di, 4)));
  r = Set2<0>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<0>(r)))));

  o = RebindMask(df, Eq(And(Add(q, Set(di, 2)), Set(di, 4)), Set(di, 4)));
  r = Set2<1>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<1>(r)))));

  o = Gt(Abs(d), Set(df, 1e+7f));
  r = Set2<0>(r, BitCast(df, IfThenZeroElse(RebindMask(du, o), BitCast(du, Get2<0>(r)))));
  r = Set2<1>(r, BitCast(df, IfThenZeroElse(RebindMask(du, o), BitCast(du, Get2<1>(r)))));
  
  o = IsInf(d);
  r = Set2<0>(r, BitCast(df, IfThenElse(RebindMask(du, o), Set(du, -1), BitCast(du, Get2<0>(r)))));
  r = Set2<1>(r, BitCast(df, IfThenElse(RebindMask(du, o), Set(du, -1), BitCast(du, Get2<1>(r)))));

  return r;
}

// Computes sin(x*pi) and cos(x*pi) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1537 XSINCOSPIF_U35
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) SinCosPiFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Mask<D> o;
  Vec<D> u, s, t, rx, ry;
  Vec2<D> r;

  u = Mul(d, Set(df, 4));
  Vec<RebindToSigned<D>> q = ConvertTo(di, u);
  q = And(Add(q, Xor(BitCast(di, ShiftRight<31>(BitCast(du, q))), Set(di, 1))), Set(di, ~1));
  s = Sub(u, ConvertTo(df, q));

  t = s;
  s = Mul(s, s);
  
  

  u = Set(df, -0.3600925265e-4);
  u = MulAdd(u, s, Set(df, +0.2490088111e-2));
  u = MulAdd(u, s, Set(df, -0.8074551076e-1));
  u = MulAdd(u, s, Set(df, +0.7853981853e+0));

  rx = Mul(u, t);

  
  
  u = Set(df, +0.3539815225e-5);
  u = MulAdd(u, s, Set(df, -0.3259574005e-3));
  u = MulAdd(u, s, Set(df, +0.1585431583e-1));
  u = MulAdd(u, s, Set(df, -0.3084251285e+0));
  u = MulAdd(u, s, Set(df, 1));

  ry = u;

  

  o = RebindMask(df, Eq(And(q, Set(di, 2)), Set(di, 0)));
  r = Create2(df, IfThenElse(o, rx, ry), IfThenElse(o, ry, rx));

  o = RebindMask(df, Eq(And(q, Set(di, 4)), Set(di, 4)));
  r = Set2<0>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<0>(r)))));

  o = RebindMask(df, Eq(And(Add(q, Set(di, 2)), Set(di, 4)), Set(di, 4)));
  r = Set2<1>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<1>(r)))));

  o = Gt(Abs(d), Set(df, 1e+7f));
  r = Set2<0>(r, BitCast(df, IfThenZeroElse(RebindMask(du, o), BitCast(du, Get2<0>(r)))));
  r = Set2<1>(r, BitCast(df, IfThenZeroElse(RebindMask(du, o), BitCast(du, Get2<1>(r)))));
  
  o = IsInf(d);
  r = Set2<0>(r, BitCast(df, IfThenElse(RebindMask(du, o), Set(du, -1), BitCast(du, Get2<0>(r)))));
  r = Set2<1>(r, BitCast(df, IfThenElse(RebindMask(du, o), Set(du, -1), BitCast(du, Get2<1>(r)))));

  return r;
}

// Computes sin(x*pi) with max(0.506 ULP, FLT_MIN) accuracy (in practice staying below 2 ULP even for subnormals)
// Translated from libm/sleefsimdsp.c:3289 xsinpif_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) SinPi(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  
  Vec2<D> x = SinPiDF(df, d);
  Vec<D> r = Add(Get2<0>(x), Get2<1>(x));

  r = IfThenElse(Eq(d, Set(df, -0.0)), Set(df, -0.0), r);
  r = BitCast(df, IfThenZeroElse(RebindMask(du, Gt(Abs(d), Set(df, TrigRangeMax4f))), BitCast(du, r)));
  r = BitCast(df, IfThenElse(RebindMask(du, IsInf(d)), Set(du, -1), BitCast(du, r)));
  
  return r;
}

// Computes cos(x*pi) with max(0.506 ULP, FLT_MIN) accuracy (in practice staying below 2 ULP even for subnormals)
// Translated from libm/sleefsimdsp.c:3339 xcospif_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) CosPi(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  
  Vec2<D> x = CosPiDF(df, d);
  Vec<D> r = Add(Get2<0>(x), Get2<1>(x));

  r = IfThenElse(Gt(Abs(d), Set(df, TrigRangeMax4f)), Set(df, 1), r);
  r = BitCast(df, IfThenElse(RebindMask(du, IsInf(d)), Set(du, -1), BitCast(du, r)));
  
  return r;
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

// Computes atan(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1743 xatanf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) AtanFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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

  t = BitCast(df, Xor(IfThenElseZero(RebindMask(du, Eq(And(q, Set(di, 2)), Set(di, 2))), BitCast(du, Set(df, -0.0f))), BitCast(du, t)));

#if HWY_ARCH_ARM && HWY_TARGET >= HWY_NEON
  t = IfThenElse(IsInf(d), MulSignBit(df, Set(df, 1.5874010519681994747517056f), d), t);
#endif

  return t;
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

// Computes atan(y/x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:1911 xatan2f_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Atan2(const D df, Vec<D> y, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Mask<D> o = Lt(Abs(x), Set(df, 2.9387372783541830947e-39f)); // nexttowardf((1.0 / FLT_MAX), 1)
  x = IfThenElse(o, Mul(x, Set(df, 1 << 24)), x);
  y = IfThenElse(o, Mul(y, Set(df, 1 << 24)), y);
  
  Vec2<D> d = ATan2DF(df, Create2(df, Abs(y), Set(df, 0)), Create2(df, x, Set(df, 0)));
  Vec<D> r = Add(Get2<0>(d), Get2<1>(d));

  r = MulSignBit(df, r, x);
  r = IfThenElse(Or(IsInf(x), Eq(x, Set(df, 0))), Sub(Set(df, Pi/2), IfInfThenElseZero(df, x, MulSignBit(df, Set(df, Pi/2), x))), r);
  r = IfThenElse(IsInf(y), Sub(Set(df, Pi/2), IfInfThenElseZero(df, x, MulSignBit(df, Set(df, Pi/4), x))), r);
  r = IfThenElse(Eq(y, Set(df, 0.0f)), BitCast(df, IfThenElseZero(RebindMask(du, SignBitMask(df, x)), BitCast(du, Set(df, (float)Pi)))), r);

  r = BitCast(df, IfThenElse(RebindMask(du, Or(IsNaN(x), IsNaN(y))), Set(du, -1), BitCast(du, MulSignBit(df, r, y))));
  return r;
}

// Computes atan(y/x) with 3.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:1818 xatan2f
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Atan2Fast(const D df, Vec<D> y, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> r = Atan2Helper(df, Abs(y), x);

  r = MulSignBit(df, r, x);
  r = IfThenElse(Or(IsInf(x), Eq(x, Set(df, 0.0f))), Sub(Set(df, (float)(Pi/2)), IfInfThenElseZero(df, x, MulSignBit(df, Set(df, (float)(Pi/2)), x))), r);
  r = IfThenElse(IsInf(y), Sub(Set(df, (float)(Pi/2)), IfInfThenElseZero(df, x, MulSignBit(df, Set(df, (float)(Pi/4)), x))), r);

  r = IfThenElse(Eq(y, Set(df, 0.0f)), BitCast(df, IfThenElseZero(RebindMask(du, SignBitMask(df, x)), BitCast(du, Set(df, (float)Pi)))), r);

  r = BitCast(df, IfThenElse(RebindMask(du, Or(IsNaN(x), IsNaN(y))), Set(du, -1), BitCast(du, MulSignBit(df, r, y))));
  return r;
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

// Computes erf(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:3471 xerff_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Erf(const D df, Vec<D> a) {
  Vec<D> t, x = Abs(a);
  Vec2<D> t2;
  Vec<D> x2 = Mul(x, x), x4 = Mul(x2, x2);
  Mask<D> o25 = Le(x, Set(df, 2.5));

  if (HWY_LIKELY(AllTrue(df, o25))) {
    // Abramowitz and Stegun
    t = Estrin(x, x2, x4, Set(df, +0.1459901541e-3), Set(df, +0.2395523916e-3), Set(df, +0.9808536561e-4), Set(df, -0.3045156700e-4), Set(df, +0.6867515367e-5), Set(df, -0.4360447008e-6));
    t2 = Poly4DF(df, x, t,
		 Create2(df, Set(df, 0.0092883445322513580322), Set(df, -2.7863745897025330755e-11)),
		 Create2(df, Set(df, 0.042275499552488327026), Set(df, 1.3461399289988106057e-09)),
		 Create2(df, Set(df, 0.070523701608180999756), Set(df, -3.6616309318707365163e-09)));
    t2 = AddFastDF(df, Set(df, 1), MulDF(df, t2, x));
    t2 = SquareDF(df, t2);
    t2 = SquareDF(df, t2);
    t2 = SquareDF(df, t2);
    t2 = SquareDF(df, t2);
    t2 = RecDF(df, t2);
  } else {
    t = Estrin(x, x2, x4, IfThenElse(o25, Set(df, +0.1459901541e-3), Set(df, +0.2708637156e-1)), IfThenElse(o25, Set(df, +0.2395523916e-3), Set(df, -0.5131045356e-2)), IfThenElse(o25, Set(df, +0.9808536561e-4), Set(df, +0.7172692567e-3)), IfThenElse(o25, Set(df, -0.3045156700e-4), Set(df, -0.6928304356e-4)), IfThenElse(o25, Set(df, +0.6867515367e-5), Set(df, +0.4115272986e-5)), IfThenElse(o25, Set(df, -0.4360447008e-6), Set(df, -0.1130012848e-6)));
    t2 = Poly4DF(df, x, t,
		 IfThenElse(df, o25, Create2(df, Set(df, 0.0092883445322513580322), Set(df, -2.7863745897025330755e-11)),
				     Create2(df, Set(df, -0.11064319312572479248), Set(df, 3.7050452777225283007e-09))),
		 IfThenElse(df, o25, Create2(df, Set(df, 0.042275499552488327026), Set(df, 1.3461399289988106057e-09)),
				     Create2(df, Set(df, -0.63192230463027954102), Set(df, -2.0200432585073177859e-08))),
		 IfThenElse(df, o25, Create2(df, Set(df, 0.070523701608180999756), Set(df, -3.6616309318707365163e-09)),
				     Create2(df, Set(df, -1.1296638250350952148), Set(df, 2.5515120196453259252e-08))));
    t2 = MulDF(df, t2, x);
    Vec2<D> s2 = AddFastDF(df, Set(df, 1), t2);
    s2 = SquareDF(df, s2);
    s2 = SquareDF(df, s2);
    s2 = SquareDF(df, s2);
    s2 = SquareDF(df, s2);
    s2 = RecDF(df, s2);
    t2 = IfThenElse(df, o25, s2, Create2(df, ExpDF_float(df, t2), Set(df, 0)));
  }

  t2 = AddDF(df, t2, Set(df, -1));
  t2 = IfThenElse(df, Lt(x, Set(df, 1e-4)), MulDF(df, Create2(df, Set(df, -1.1283792257308959961), Set(df, 5.8635383422197591097e-08)), x), t2);

  Vec<D> z = Neg(Add(Get2<0>(t2), Get2<1>(t2)));
  z = IfThenElse(Ge(x, Set(df, 6)), Set(df, 1), z);
  z = IfThenElse(IsInf(a), Set(df, 1), z);
  z = IfThenElse(Eq(a, Set(df, 0)), Set(df, 0), z);
  z = MulSignBit(df, z, a);

  return z;
}

// Computes 1 - erf(x) with 1.5 ULP accuracy
// Translated from libm/sleefsimdsp.c:3536 xerfcf_u15
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Erfc(const D df, Vec<D> a) {
  Vec<D> s = a, r = Set(df, 0), t;
  Vec2<D> u, d, x;
  a = Abs(a);
  Mask<D> o0 = Lt(a, Set(df, 1.0));
  Mask<D> o1 = Lt(a, Set(df, 2.2));
  Mask<D> o2 = Lt(a, Set(df, 4.3));
  Mask<D> o3 = Lt(a, Set(df, 10.1));

  u = IfThenElse(df, o1, Create2(df, a, Set(df, 0)), DivDF(df, Create2(df, Set(df, 1), Set(df, 0)), Create2(df, a, Set(df, 0))));

  t = IfThenElse4(df, o0, o1, o2, -0.8638041618e-4f, -0.6236977242e-5f, -0.3869504035e+0f, +0.1115344167e+1f);
  t = MulAdd(t, Get2<0>(u), IfThenElse4(df, o0, o1, o2, +0.6000166177e-3f, +0.5749821503e-4f, +0.1288077235e+1f, -0.9454904199e+0f));
  t = MulAdd(t, Get2<0>(u), IfThenElse4(df, o0, o1, o2, -0.1665703603e-2f, +0.6002851478e-5f, -0.1816803217e+1f, -0.3667259514e+0f));
  t = MulAdd(t, Get2<0>(u), IfThenElse4(df, o0, o1, o2, +0.1795156277e-3f, -0.2851036377e-2f, +0.1249150872e+1f, +0.7155663371e+0f));
  t = MulAdd(t, Get2<0>(u), IfThenElse4(df, o0, o1, o2, +0.1914106123e-1f, +0.2260518074e-1f, -0.1328857988e+0f, -0.1262947265e-1f));

  d = MulDF(df, u, t);
  d = AddDF(df, d, IfThenElse4(df, o0, o1, o2, -0.102775359343930288081655368891e+0, -0.105247583459338632253369014063e+0, -0.482365310333045318680618892669e+0, -0.498961546254537647970305302739e+0));
  d = MulDF(df, d, u);
  d = AddDF(df, d, IfThenElse4(df, o0, o1, o2, -0.636619483208481931303752546439e+0, -0.635609463574589034216723775292e+0, -0.134450203224533979217859332703e-2, -0.471199543422848492080722832666e-4));
  d = MulDF(df, d, u);
  d = AddDF(df, d, IfThenElse4(df, o0, o1, o2, -0.112837917790537404939545770596e+1, -0.112855987376668622084547028949e+1, -0.572319781150472949561786101080e+0, -0.572364030327966044425932623525e+0));
  
  x = MulDF(df, IfThenElse(df, o1, d, Create2(df, Neg(a), Set(df, 0))), a);
  x = IfThenElse(df, o1, x, AddDF(df, x, d));

  x = ExpDF(df, x);
  x = IfThenElse(df, o1, x, MulDF(df, x, u));

  r = IfThenElse(o3, Add(Get2<0>(x), Get2<1>(x)), Set(df, 0));
  r = IfThenElse(SignBitMask(df, s), Sub(Set(df, 2), r), r);
  r = IfThenElse(IsNaN(s), Set(df, NanFloat), r);
  return r;
}

// Computes gamma(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:3428 xtgammaf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Gamma(const D df, Vec<D> a) {
  Vec4<D> d = GammaQF(df, a);
  Vec2<D> y = MulDF(df, ExpDF(df, Create2(df, Get4<0>(d), Get4<1>(d))), Create2(df, Get4<2>(d), Get4<3>(d)));
  Vec<D> r = Add(Get2<0>(y), Get2<1>(y));
  Mask<D> o;

  o = Or(Or(Eq(a, Set(df, -InfFloat)), And(Lt(a, Set(df, 0)), IsInt(df, a))), And(And(IsFinite(a), Lt(a, Set(df, 0))), IsNaN(r)));
  r = IfThenElse(o, Set(df, NanFloat), r);

  o = And(And(Or(Eq(a, Set(df, InfFloat)), IsFinite(a)), Ge(a, Set(df, -FloatMin))), Or(Or(Eq(a, Set(df, 0)), Gt(a, Set(df, 36))), IsNaN(r)));
  r = IfThenElse(o, MulSignBit(df, Set(df, InfFloat), a), r);
  
  return r;
}

// Computes log(gamma(x)) with 1.0 ULP accuracy
// Translated from libm/sleefsimdsp.c:3447 xlgammaf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) LogGamma(const D df, Vec<D> a) {
  Vec4<D> d = GammaQF(df, a);
  Vec2<D> y = AddDF(df, Create2(df, Get4<0>(d), Get4<1>(d)), LogFastDF(df, AbsDF(df, Create2(df, Get4<2>(d), Get4<3>(d)))));
  Vec<D> r = Add(Get2<0>(y), Get2<1>(y));
  Mask<D> o;

  o = Or(IsInf(a), Or(And(Le(a, Set(df, 0)), IsInt(df, a)), And(IsFinite(a), IsNaN(r))));
  r = IfThenElse(o, Set(df, InfFloat), r);

  return r;
}

// Computes fmod(x), the floating point remainder
// Translated from libm/sleefsimdsp.c:3173 xfmodf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Fmod(const D df, Vec<D> x, Vec<D> y) {
  Vec<D> nu = Abs(x), de = Abs(y), s = Set(df, 1), q;
  Mask<D> o = Lt(de, Set(df, FloatMin));
  nu = IfThenElse(o, Mul(nu, Set(df, UINT64_C(1) << 25)), nu);
  de = IfThenElse(o, Mul(de, Set(df, UINT64_C(1) << 25)), de);
  s  = IfThenElse(o, Mul(s, Set(df, 1.0f / (UINT64_C(1) << 25))), s);
  Vec<D> rde = Toward0(df, Div(Set(df, 1.0), de));
#if HWY_ARCH_ARM && HWY_TARGET >= HWY_NEON
  rde = Toward0(df, rde);
#endif
  Vec2<D> r = Create2(df, nu, Set(df, 0));

  for(int i=0;i<8;i++) { // ceil(log2(FLT_MAX) / 22)+1
    q = Trunc(Mul(Toward0(df, Get2<0>(r)), rde));
    q = IfThenElse(And(Gt(Mul(Set(df, 3), de), Get2<0>(r)), Ge(Get2<0>(r), de)), Set(df, 2), q);
    q = IfThenElse(And(Gt(Mul(Set(df, 2), de), Get2<0>(r)), Ge(Get2<0>(r), de)), Set(df, 1), q);
    r = NormalizeDF(df, AddDF(df, r, MulDF(df, Trunc(q), Neg(de))));
    if (AllTrue(df, Lt(Get2<0>(r), de))) break;
  }
  
  Vec<D> ret = Mul(Add(Get2<0>(r), Get2<1>(r)), s);
  ret = IfThenElse(Eq(Add(Get2<0>(r), Get2<1>(r)), de), Set(df, 0), ret);

  ret = MulSignBit(df, ret, x);

  ret = IfThenElse(Lt(nu, de), x, ret);
  ret = IfThenElse(Eq(de, Set(df, 0)), Set(df, NanFloat), ret);

  return ret;
}

// Computes remainder(x), the signed floating point remainder
// Translated from libm/sleefsimdsp.c:3218 xremainderf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) Remainder(const D df, Vec<D> x, Vec<D> y) {
  RebindToSigned<D> di;
  
  Vec<D> n = Abs(x), d = Abs(y), s = Set(df, 1), q;
  Mask<D> o = Lt(d, Set(df, FloatMin*2));
  n = IfThenElse(o, Mul(n, Set(df, UINT64_C(1) << 25)), n);
  d = IfThenElse(o, Mul(d, Set(df, UINT64_C(1) << 25)), d);
  s  = IfThenElse(o, Mul(s, Set(df, 1.0f / (UINT64_C(1) << 25))), s);
  Vec2<D> r = Create2(df, n, Set(df, 0));
  Vec<D> rd = Div(Set(df, 1.0), d);
  Mask<D> qisodd = Ne(Set(df, 0), Set(df, 0));

  for(int i=0;i<8;i++) { // ceil(log2(FLT_MAX) / 22)+1
    q = Round(Mul(Get2<0>(r), rd));
    q = IfThenElse(Lt(Abs(Get2<0>(r)), Mul(d, Set(df, 1.5f))), MulSignBit(df, Set(df, 1.0f), Get2<0>(r)), q);
    q = IfThenElse(Or(Lt(Abs(Get2<0>(r)), Mul(d, Set(df, 0.5f))), AndNot(qisodd, Eq(Abs(Get2<0>(r)), Mul(d, Set(df, 0.5f))))), Set(df, 0.0), q);
    if (AllTrue(df, Eq(q, Set(df, 0)))) break;
    q = IfThenElse(IsInf(Mul(q, Neg(d))), Add(q, MulSignBit(df, Set(df, -1), Get2<0>(r))), q);
    qisodd = Xor(qisodd, And(RebindMask(df, Eq(And(ConvertTo(di, q), Set(di, 1)), Set(di, 1))), Lt(Abs(q), Set(df, 1 << 24))));
    r = NormalizeDF(df, AddDF(df, r, MulDF(df, q, Neg(d))));
  }
  
  Vec<D> ret = Mul(Add(Get2<0>(r), Get2<1>(r)), s);
  ret = MulSignBit(df, ret, x);
  ret = IfThenElse(IsInf(y), IfThenElse(IsInf(x), Set(df, NanFloat), x), ret);
  ret = IfThenElse(Eq(d, Set(df, 0)), Set(df, NanFloat), ret);
  return ret;
}

// Computes x * 2^exp
// Translated from libm/sleefsimdsp.c:543 xldexpf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) LdExp(const D df, Vec<D> x, Vec<RebindToSigned<D>> q) {
 return LoadExp(df, x, q); }

// Decomposes x into 2^exp * fr where abs(fr) is in [0.5, 1), returning fr
// Translated from libm/sleefsimdsp.c:3128 xfrfrexpf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) FrFrexp(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  x = IfThenElse(Lt(Abs(x), Set(df, FloatMin)), Mul(x, Set(df, UINT64_C(1) << 30)), x);

  Vec<RebindToUnsigned<D>> xm = BitCast(du, x);
  xm = And(xm, Set(du, (static_cast<uint64_t>(~0x7f800000U) << 32) | ~0x7f800000U));
  xm = Or(xm, Set(du, (static_cast<uint64_t>(0x3f000000U) << 32) | 0x3f000000U));

  Vec<D> ret = BitCast(df, xm);

  ret = IfThenElse(IsInf(x), MulSignBit(df, Set(df, InfFloat), x), ret);
  ret = IfThenElse(Eq(x, Set(df, 0)), x, ret);
  
  return ret;
}

// Decomposes x into 2^exp * fr where abs(fr) is in [0.5, 1), returning exp
// Translated from libm/sleefsimdsp.c:3144 xexpfrexpf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<RebindToSigned<D>>) ExpFrexp(const D df, Vec<D> x) {
  RebindToSigned<D> di;
  
  
  return Set(di, 0);
}

// Computes the unbiased exponent of x
// Translated from libm/sleefsimdsp.c:508 xilogbf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<RebindToSigned<D>>) ILogB(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec<RebindToSigned<D>> e = ILogB1(df, Abs(d));
  e = IfThenElse(RebindMask(di, Eq(d, Set(df, 0.0f))), Set(di, ILogB0), e);
  e = IfThenElse(RebindMask(di, IsNaN(d)), Set(di, ILogBNan), e);
  e = IfThenElse(RebindMask(di, IsInf(d)), Set(di, IntMax), e);
  return e;
}

// Decompose x into an integer and fractional part
// Translated from libm/sleefsimdsp.c:1591 XMODFF
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec2<D>) Modf(const D df, Vec<D> x) {
  RebindToSigned<D> di;
  
  Vec<D> fr = Sub(x, ConvertTo(df, ConvertTo(di, x)));
  fr = IfThenElse(Gt(Abs(x), Set(df, INT64_C(1) << 23)), Set(df, 0), fr);

  Vec2<D> ret;

  ret = Create2(df, CopySign(df, fr, x), CopySign(df, Sub(x, fr), x));

  return ret;
}

// Returns the next representable value after x in the direction of y
// Translated from libm/sleefsimdsp.c:3105 xnextafterf
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) NextAfter(const D df, Vec<D> x, Vec<D> y) {
  RebindToSigned<D> di;
  
  x = IfThenElse(Eq(x, Set(df, 0)), MulSignBit(df, Set(df, 0), y), x);
  Vec<RebindToSigned<D>> xi2 = BitCast(di, x);
  Mask<D> c = Xor(SignBitMask(df, x), Ge(y, x));

  xi2 = IfThenElse(RebindMask(di, c), Sub(Set(di, 0), Xor(xi2, Set(di, (int)(1U << 31)))), xi2);

  xi2 = IfThenElse(RebindMask(di, Ne(x, y)), Sub(xi2, Set(di, 1)), xi2);

  xi2 = IfThenElse(RebindMask(di, c), Sub(Set(di, 0), Xor(xi2, Set(di, (int)(1U << 31)))), xi2);

  Vec<D> ret = BitCast(df, xi2);

  ret = IfThenElse(And(Eq(ret, Set(df, 0)), Ne(x, Set(df, 0))), MulSignBit(df, Set(df, 0), x), ret);

  ret = IfThenElse(And(Eq(x, Set(df, 0)), Eq(y, Set(df, 0))), y, ret);

  ret = IfThenElse(Or(IsNaN(x), IsNaN(y)), Set(df, NanFloat), ret);
  
  return ret;
}

// Computes sin(pi) very quickly in range [-30, 30] with max(2e-6, 350 ULP) accuracy. Falls back to SinFast out of range.
// Translated from libm/sleefsimdsp.c:1165 xfastsinf_u3500
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) SinFaster(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<RebindToSigned<D>> q;
  Vec<D> u, s, t = d;

  s = Mul(d, Set(df, (float)OneOverPi));
  u = Round(s);
  q = NearestInt(s);
  d = MulAdd(u, Set(df, -(float)Pi), d);

  s = Mul(d, d);

  u = Set(df, -0.1881748176e-3);
  u = MulAdd(u, s, Set(df, +0.8323502727e-2));
  u = MulAdd(u, s, Set(df, -0.1666651368e+0));
  u = MulAdd(Mul(s, d), u, d);

  u = BitCast(df, Xor(IfThenElseZero(RebindMask(du, Eq(And(q, Set(di, 1)), Set(di, 1))), BitCast(du, Set(df, -0.0f))), BitCast(du, u)));

  Mask<D> g = Lt(Abs(t), Set(df, 30.0f));
  if (!HWY_LIKELY(AllTrue(df, g))) return IfThenElse(g, u, sleef::SinFast(df, t));

  return u;
}

// Computes cos(pi) very quickly in range [-30, 30] with max(2e-6, 350 ULP) accuracy. Falls back to CosFast out of range.
// Translated from libm/sleefsimdsp.c:1189 xfastcosf_u3500
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) CosFaster(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  Vec<RebindToSigned<D>> q;
  Vec<D> u, s, t = d;

  s = MulAdd(d, Set(df, (float)OneOverPi), Set(df, -0.5f));
  u = Round(s);
  q = NearestInt(s);
  d = MulAdd(u, Set(df, -(float)Pi), Sub(d, Set(df, (float)Pi * 0.5f)));

  s = Mul(d, d);

  u = Set(df, -0.1881748176e-3);
  u = MulAdd(u, s, Set(df, +0.8323502727e-2));
  u = MulAdd(u, s, Set(df, -0.1666651368e+0));
  u = MulAdd(Mul(s, d), u, d);

  u = BitCast(df, Xor(IfThenElseZero(RebindMask(du, Eq(And(q, Set(di, 1)), Set(di, 0))), BitCast(du, Set(df, -0.0f))), BitCast(du, u)));

  Mask<D> g = Lt(Abs(t), Set(df, 30.0f));
  if (!HWY_LIKELY(AllTrue(df, g))) return IfThenElse(g, u, sleef::CosFast(df, t));

  return u;
}

// Computes x^y very quickly with 350 ULP accuracy
// Translated from libm/sleefsimdsp.c:2403 xfastpowf_u3500
template<class D>
HWY_INLINE HWY_SLEEF_IF_FLOAT(D, Vec<D>) PowFaster(const D df, Vec<D> x, Vec<D> y) {
  RebindToSigned<D> di;
  
  Vec<D> result = ExpFast(df, Mul(LogFast(df, Abs(x)), y));
  Mask<D> yisint = Or(Eq(Trunc(y), y), Gt(Abs(y), Set(df, 1 << 24)));
  Mask<D> yisodd = And(And(RebindMask(df, Eq(And(ConvertTo(di, y), Set(di, 1)), Set(di, 1))), yisint), Lt(Abs(y), Set(df, 1 << 24)));

  result = IfThenElse(And(SignBitMask(df, x), yisodd), Neg(result), result);

  result = IfThenElse(Eq(x, Set(df, 0)), Set(df, 0), result);
  result = IfThenElse(Eq(y, Set(df, 0)), Set(df, 1), result);

  return result;
}

// Computes e^x
// Translated from libm/sleefsimddp.c:2146 xexp
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Exp(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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
#else 
  Vec<D> s2 = Mul(s, s), s4 = Mul(s2, s2), s8 = Mul(s4, s4);
  u = Estrin(s, s2, s4, s8, Set(df, 0.166666666666666851703837), Set(df, 0.0416666666666665047591422), Set(df, 0.00833333333331652721664984), Set(df, 0.00138888888889774492207962), Set(df, 0.000198412698960509205564975), Set(df, 2.4801587159235472998791e-05), Set(df, 2.75572362911928827629423e-06), Set(df, 2.75573911234900471893338e-07), Set(df, 2.51112930892876518610661e-08), Set(df, 2.08860621107283687536341e-09));
  u = MulAdd(u, s, Set(df, +0.5000000000000000000e+0));

  u = Add(Set(df, 1), MulAdd(Mul(s, s), u, s));
#endif 
  
  u = LoadExp2(df, u, q);

  u = IfThenElse(Gt(d, Set(df, 709.78271289338399673222338991)), Set(df, InfDouble), u);
  u = BitCast(df, IfThenZeroElse(RebindMask(du, Lt(d, Set(df, -1000))), BitCast(du, u)));

  return u;
}

// Computes 2^x
// Translated from libm/sleefsimddp.c:2686 xexp2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Exp2(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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

// Computes sqrt(x) with 0.5001 ULP accuracy
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
  e = Add(ILogB1(df, Abs(d)), Set(di, 1));
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
  e = Add(ILogB1(df, Abs(d)), Set(di, 1));
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

// Computes sqrt(x^2 + y^2) with 0.5001 ULP accuracy
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
  result = IfThenElse(Gt(Get2<0>(d), Set(df, 709.78271289338399673222338991)), Set(df, InfDouble), result);

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
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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
      Vec3<D> ddi = PayneHanekReduction(df, d);
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

  u = BitCast(df, Xor(IfThenElseZero(RebindMask(du, Eq(And(ql, Set(di, 1)), Set(di, 1))), BitCast(du, Set(df, -0.0))), BitCast(du, u)));

  u = IfThenElse(Eq(d, Set(df, 0)), d, u);
  
  return u; 
}

// Computes cos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:787 xcos_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Cos(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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
      Vec3<D> ddi = PayneHanekReduction(df, d);
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
  
  u = BitCast(df, Xor(IfThenElseZero(RebindMask(du, Eq(And(ql, Set(di, 2)), Set(di, 0))), BitCast(du, Set(df, -0.0))), BitCast(du, u)));
  
  return u; 
}

// Computes tan(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:1645 xtan_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Tan(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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
    x = AddDD(df, MulDD(df, Create2(df, Set(df, TwoOverPiHi), Set(df, TwoOverPiLo)), d),
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
      Vec3<D> ddi = PayneHanekReduction(df, d);
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

  return u; 
}

// Computes sin(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:382 xsin
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) SinFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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
      Vec3<D> ddi = PayneHanekReduction(df, r);
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

  d = BitCast(df, Xor(IfThenElseZero(RebindMask(du, Eq(And(ql, Set(di, 1)), Set(di, 1))), BitCast(du, Set(df, -0.0))), BitCast(du, d)));

  Vec<D> s2 = Mul(s, s), s4 = Mul(s2, s2);
  u = Estrin(s, s2, s4, Set(df, 0.00833333333333332974823815), Set(df, -0.000198412698412696162806809), Set(df, 2.75573192239198747630416e-06), Set(df, -2.50521083763502045810755e-08), Set(df, 1.60590430605664501629054e-10), Set(df, -7.64712219118158833288484e-13), Set(df, 2.81009972710863200091251e-15), Set(df, -7.97255955009037868891952e-18));
  u = MulAdd(u, s, Set(df, -0.166666666666666657414808));

  u = Add(Mul(s, Mul(u, d)), d);

  u = IfThenElse(Eq(r, Set(df, -0.0)), r, u);
  
  return u; 
}

// Computes cos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:652 xcos
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) CosFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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
      Vec3<D> ddi = PayneHanekReduction(df, r);
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

  d = BitCast(df, Xor(IfThenElseZero(RebindMask(du, Eq(And(ql, Set(di, 2)), Set(di, 0))), BitCast(du, Set(df, -0.0))), BitCast(du, d)));

  Vec<D> s2 = Mul(s, s), s4 = Mul(s2, s2);
  u = Estrin(s, s2, s4, Set(df, 0.00833333333333332974823815), Set(df, -0.000198412698412696162806809), Set(df, 2.75573192239198747630416e-06), Set(df, -2.50521083763502045810755e-08), Set(df, 1.60590430605664501629054e-10), Set(df, -7.64712219118158833288484e-13), Set(df, 2.81009972710863200091251e-15), Set(df, -7.97255955009037868891952e-18));
  u = MulAdd(u, s, Set(df, -0.166666666666666657414808));

  u = Add(Mul(s, Mul(u, d)), d);
  
  return u; 
}

// Computes tan(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:1517 xtan
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) TanFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
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
      Vec3<D> ddi = PayneHanekReduction(df, d);
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
  
  return u; 
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

// Computes atan(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:2042 xatan_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Atan(const D df, Vec<D> d) {
  Vec2<D> d2 = ATan2DD(df, Create2(df, Abs(d), Set(df, 0)), Create2(df, Set(df, 1), Set(df, 0)));
  Vec<D> r = Add(Get2<0>(d2), Get2<1>(d2));
  r = IfThenElse(IsInf(d), Set(df, 1.570796326794896557998982), r);
  return MulSignBit(df, r, d);
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
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Vec<D> t, u;
  Vec<RebindToSigned<D>> q;


  q = IfNegThenElseZero(df, s, Set(di, 2));
  s = Abs(s);

  q = IfLtThenElseZero(df, Set(df, 1), s, Add(q, Set(di, 1)), q);
  s = IfThenElse(Lt(Set(df, 1), s), Div(Set(df, 1.0), s), s);

  t = Mul(s, s);

  Vec<D> t2 = Mul(t, t), t4 = Mul(t2, t2), t8 = Mul(t4, t4), t16 = Mul(t8, t8);
  u = Estrin(t, t2, t4, t8, t16, Set(df, -0.333333333333311110369124), Set(df, 0.199999999996591265594148), Set(df, -0.14285714266771329383765), Set(df, 0.111111105648261418443745), Set(df, -0.090908995008245008229153), Set(df, 0.0769219538311769618355029), Set(df, -0.0666573579361080525984562), Set(df, 0.0587666392926673580854313), Set(df, -0.0523674852303482457616113), Set(df, 0.0466667150077840625632675), Set(df, -0.0407629191276836500001934), Set(df, 0.0337852580001353069993897), Set(df, -0.0254517624932312641616861), Set(df, 0.016599329773529201970117), Set(df, -0.00889896195887655491740809), Set(df, 0.00370026744188713119232403), Set(df, -0.00110611831486672482563471), Set(df, 0.000209850076645816976906797), Set(df, -1.88796008463073496563746e-05));
  
  t = MulAdd(s, Mul(t, u), s);

  t = IfThenElse(RebindMask(df, Eq(And(q, Set(di, 1)), Set(di, 1))), Sub(Set(df, Pi/2), t), t);
  t = BitCast(df, Xor(IfThenElseZero(RebindMask(du, Eq(And(q, Set(di, 2)), Set(di, 2))), BitCast(du, Set(df, -0.0))), BitCast(du, t)));



  return t;
}

// Computes atan(y/x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:1902 xatan2_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Atan2(const D df, Vec<D> y, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Mask<D> o = Lt(Abs(x), Set(df, 5.5626846462680083984e-309)); // nexttoward((1.0 / DBL_MAX), 1)
  x = IfThenElse(o, Mul(x, Set(df, UINT64_C(1) << 53)), x);
  y = IfThenElse(o, Mul(y, Set(df, UINT64_C(1) << 53)), y);

  Vec2<D> d = ATan2DD(df, Create2(df, Abs(y), Set(df, 0)), Create2(df, x, Set(df, 0)));
  Vec<D> r = Add(Get2<0>(d), Get2<1>(d));

  r = MulSignBit(df, r, x);
  r = IfThenElse(Or(IsInf(x), Eq(x, Set(df, 0))), Sub(Set(df, Pi/2), IfInfThenElseZero(df, x, MulSignBit(df, Set(df, Pi/2), x))), r);
  r = IfThenElse(IsInf(y), Sub(Set(df, Pi/2), IfInfThenElseZero(df, x, MulSignBit(df, Set(df, Pi/4), x))), r);
  r = IfThenElse(Eq(y, Set(df, 0.0)), BitCast(df, IfThenElseZero(RebindMask(du, SignBitMask(df, x)), BitCast(du, Set(df, Pi)))), r);

  r = BitCast(df, IfThenElse(RebindMask(du, Or(IsNaN(x), IsNaN(y))), Set(du, -1), BitCast(du, MulSignBit(df, r, y))));
  return r;
}

// Computes atan(y/x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:1890 xatan2
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Atan2Fast(const D df, Vec<D> y, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  Vec<D> r = Atan2Helper(df, Abs(y), x);

  r = MulSignBit(df, r, x);
  r = IfThenElse(Or(IsInf(x), Eq(x, Set(df, 0))), Sub(Set(df, Pi/2), IfInfThenElseZero(df, x, MulSignBit(df, Set(df, Pi/2), x))), r);
  r = IfThenElse(IsInf(y), Sub(Set(df, Pi/2), IfInfThenElseZero(df, x, MulSignBit(df, Set(df, Pi/4), x))), r);
  r = IfThenElse(Eq(y, Set(df, 0.0)), BitCast(df, IfThenElseZero(RebindMask(du, SignBitMask(df, x)), BitCast(du, Set(df, Pi)))), r);

  r = BitCast(df, IfThenElse(RebindMask(du, Or(IsNaN(x), IsNaN(y))), Set(du, -1), BitCast(du, MulSignBit(df, r, y))));
  return r;
}

// Computes sin(x) and cos(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:1086 XSINCOS_U1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SinCos(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Mask<D> o;
  Vec<D> u, rx, ry;
  Vec2<D> r, s, t, x;
  Vec<RebindToSigned<D>> ql;
  
  const Vec<D> dql = Round(Mul(d, Set(df, 2 * OneOverPi)));
  ql = ConvertTo(di, Round(dql));
  u = MulAdd(dql, Set(df, -PiA2*0.5), d);
  s = AddFastDD(df, u, Mul(dql, Set(df, -PiB2*0.5)));
  Mask<D> g = Lt(Abs(d), Set(df, TrigRangeMax2));

  if (!HWY_LIKELY(AllTrue(df, g))) {
    Vec<D> dqh = Trunc(Mul(d, Set(df, 2*OneOverPi / (1 << 24))));
    dqh = Mul(dqh, Set(df, 1 << 24));
    const Vec<D> dql = Round(Sub(Mul(d, Set(df, 2*OneOverPi)), dqh));
    
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
      Vec3<D> ddi = PayneHanekReduction(df, d);
      x = Create2(df, Get3<0>(ddi), Get3<1>(ddi));
      o = Or(IsInf(d), IsNaN(d));
      x = Set2<0>(x, BitCast(df, IfThenElse(RebindMask(du, o), Set(du, -1), BitCast(du, Get2<0>(x)))));
      x = Set2<1>(x, BitCast(df, IfThenElse(RebindMask(du, o), Set(du, -1), BitCast(du, Get2<1>(x)))));

      ql = IfThenElse(RebindMask(di, g), ql, BitCast(di, Get3<2>(ddi)));
      s = IfThenElse(df, g, s, x);
    }
  }
  
  t = s;

  s = Set2<0>(s, SquareDD_double(df, s));
  
  u = Set(df, 1.58938307283228937328511e-10);
  u = MulAdd(u, Get2<0>(s), Set(df, -2.50506943502539773349318e-08));
  u = MulAdd(u, Get2<0>(s), Set(df, 2.75573131776846360512547e-06));
  u = MulAdd(u, Get2<0>(s), Set(df, -0.000198412698278911770864914));
  u = MulAdd(u, Get2<0>(s), Set(df, 0.0083333333333191845961746));
  u = MulAdd(u, Get2<0>(s), Set(df, -0.166666666666666130709393));

  u = Mul(u, Mul(Get2<0>(s), Get2<0>(t)));

  x = AddFastDD(df, t, u);
  rx = Add(Get2<0>(x), Get2<1>(x));

  rx = IfThenElse(Eq(d, Set(df, -0.0)), Set(df, -0.0), rx);
  
  u = Set(df, -1.13615350239097429531523e-11);
  u = MulAdd(u, Get2<0>(s), Set(df, 2.08757471207040055479366e-09));
  u = MulAdd(u, Get2<0>(s), Set(df, -2.75573144028847567498567e-07));
  u = MulAdd(u, Get2<0>(s), Set(df, 2.48015872890001867311915e-05));
  u = MulAdd(u, Get2<0>(s), Set(df, -0.00138888888888714019282329));
  u = MulAdd(u, Get2<0>(s), Set(df, 0.0416666666666665519592062));
  u = MulAdd(u, Get2<0>(s), Set(df, -0.5));

  x = AddFastDD(df, Set(df, 1), MulDD(df, Get2<0>(s), u));
  ry = Add(Get2<0>(x), Get2<1>(x));

  o = RebindMask(df, Eq(And(ql, Set(di, 1)), Set(di, 0)));
  r = Create2(df, IfThenElse(o, rx, ry), IfThenElse(o, ry, rx));

  o = RebindMask(df, Eq(And(ql, Set(di, 2)), Set(di, 2)));
  r = Set2<0>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<0>(r)))));

  o = RebindMask(df, Eq(And(Add(ql, Set(di, 1)), Set(di, 2)), Set(di, 2)));
  r = Set2<1>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<1>(r)))));

  return r; 
}

// Computes sin(x) and cos(x) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:942 XSINCOS
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SinCosFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Mask<D> o;
  Vec<D> u, t, rx, ry, s = d;
  Vec2<D> r;
  Vec<RebindToSigned<D>> ql;

  Vec<D> dql = Round(Mul(s, Set(df, 2 * OneOverPi)));
  ql = ConvertTo(di, Round(dql));
  s = MulAdd(dql, Set(df, -PiA2 * 0.5), s);
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
    g = Lt(Abs(d), Set(df, TrigRangeMax));

    if (!HWY_LIKELY(AllTrue(df, g))) {
      Vec3<D> ddi = PayneHanekReduction(df, d);
      u = Add(Get2<0>(Create2(df, Get3<0>(ddi), Get3<1>(ddi))), Get2<1>(Create2(df, Get3<0>(ddi), Get3<1>(ddi))));
      u = BitCast(df, IfThenElse(RebindMask(du, Or(IsInf(d), IsNaN(d))), Set(du, -1), BitCast(du, u)));

      ql = IfThenElse(RebindMask(di, g), ql, BitCast(di, Get3<2>(ddi)));
      s = IfThenElse(g, s, u);
    }
  }
  
  t = s;

  s = Mul(s, s);

  u = Set(df, 1.58938307283228937328511e-10);
  u = MulAdd(u, s, Set(df, -2.50506943502539773349318e-08));
  u = MulAdd(u, s, Set(df, 2.75573131776846360512547e-06));
  u = MulAdd(u, s, Set(df, -0.000198412698278911770864914));
  u = MulAdd(u, s, Set(df, 0.0083333333333191845961746));
  u = MulAdd(u, s, Set(df, -0.166666666666666130709393));

  rx = MulAdd(Mul(u, s), t, t);
  rx = IfThenElse(Eq(d, Set(df, -0.0)), Set(df, -0.0), rx);

  u = Set(df, -1.13615350239097429531523e-11);
  u = MulAdd(u, s, Set(df, 2.08757471207040055479366e-09));
  u = MulAdd(u, s, Set(df, -2.75573144028847567498567e-07));
  u = MulAdd(u, s, Set(df, 2.48015872890001867311915e-05));
  u = MulAdd(u, s, Set(df, -0.00138888888888714019282329));
  u = MulAdd(u, s, Set(df, 0.0416666666666665519592062));
  u = MulAdd(u, s, Set(df, -0.5));

  ry = MulAdd(s, u, Set(df, 1));

  o = RebindMask(df, Eq(And(ql, Set(di, 1)), Set(di, 0)));
  r = Create2(df, IfThenElse(o, rx, ry), IfThenElse(o, ry, rx));

  o = RebindMask(df, Eq(And(ql, Set(di, 2)), Set(di, 2)));
  r = Set2<0>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<0>(r)))));

  o = RebindMask(df, Eq(And(Add(ql, Set(di, 1)), Set(di, 2)), Set(di, 2)));
  r = Set2<1>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<1>(r)))));
  
  return r; 
}

// Computes sin(x*pi) and cos(x*pi) with max(0.506 ULP, FLT_MIN) accuracy (in practice staying below 2 ULP even for subnormals)
// Translated from libm/sleefsimddp.c:1245 XSINCOSPI_U05
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SinCosPi(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Mask<D> o;
  Vec<D> u, s, t, rx, ry;
  Vec2<D> r, x, s2;

  u = Mul(d, Set(df, 4.0));
  Vec<RebindToSigned<D>> q = ConvertTo(di, Trunc(u));
  q = And(Add(q, Xor(BitCast(di, ShiftRight<63>(BitCast(du, q))), Set(di, 1))), Set(di, ~1));
  s = Sub(u, ConvertTo(df, q));
  
  t = s;
  s = Mul(s, s);
  s2 = MulDD(df, t, t);
  
  

  u = Set(df, -2.02461120785182399295868e-14);
  u = MulAdd(u, s, Set(df, 6.94821830580179461327784e-12));
  u = MulAdd(u, s, Set(df, -1.75724749952853179952664e-09));
  u = MulAdd(u, s, Set(df, 3.13361688966868392878422e-07));
  u = MulAdd(u, s, Set(df, -3.6576204182161551920361e-05));
  u = MulAdd(u, s, Set(df, 0.00249039457019271850274356));
  x = AddDD(df, Mul(u, s), Create2(df, Set(df, -0.0807455121882807852484731), Set(df, 3.61852475067037104849987e-18)));
  x = AddDD(df, MulDD(df, s2, x), Create2(df, Set(df, 0.785398163397448278999491), Set(df, 3.06287113727155002607105e-17)));

  x = MulDD(df, x, t);
  rx = Add(Get2<0>(x), Get2<1>(x));

  rx = IfThenElse(Eq(d, Set(df, -0.0)), Set(df, -0.0), rx);
  
  
  
  u = Set(df, 9.94480387626843774090208e-16);
  u = MulAdd(u, s, Set(df, -3.89796226062932799164047e-13));
  u = MulAdd(u, s, Set(df, 1.15011582539996035266901e-10));
  u = MulAdd(u, s, Set(df, -2.4611369501044697495359e-08));
  u = MulAdd(u, s, Set(df, 3.59086044859052754005062e-06));
  u = MulAdd(u, s, Set(df, -0.000325991886927389905997954));
  x = AddDD(df, Mul(u, s), Create2(df, Set(df, 0.0158543442438155018914259), Set(df, -1.04693272280631521908845e-18)));
  x = AddDD(df, MulDD(df, s2, x), Create2(df, Set(df, -0.308425137534042437259529), Set(df, -1.95698492133633550338345e-17)));

  x = AddDD(df, MulDD(df, x, s2), Set(df, 1));
  ry = Add(Get2<0>(x), Get2<1>(x));

  

  o = RebindMask(df, Eq(And(q, Set(di, 2)), Set(di, 0)));
  r = Create2(df, IfThenElse(o, rx, ry), IfThenElse(o, ry, rx));

  o = RebindMask(df, Eq(And(q, Set(di, 4)), Set(di, 4)));
  r = Set2<0>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<0>(r)))));

  o = RebindMask(df, Eq(And(Add(q, Set(di, 2)), Set(di, 4)), Set(di, 4)));
  r = Set2<1>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<1>(r)))));

  o = Gt(Abs(d), Set(df, TrigRangeMax3/4));
  r = Set2<0>(r, BitCast(df, IfThenZeroElse(RebindMask(du, o), BitCast(du, Get2<0>(r)))));
  r = Set2<1>(r, IfThenElse(o, Set(df, 1), Get2<1>(r)));

  o = IsInf(d);
  r = Set2<0>(r, BitCast(df, IfThenElse(RebindMask(du, o), Set(du, -1), BitCast(du, Get2<0>(r)))));
  r = Set2<1>(r, BitCast(df, IfThenElse(RebindMask(du, o), Set(du, -1), BitCast(du, Get2<1>(r)))));

  return r;
}

// Computes sin(x*pi) and cos(x*pi) with 3.5 ULP accuracy
// Translated from libm/sleefsimddp.c:1311 XSINCOSPI_U35
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) SinCosPiFast(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  RebindToUnsigned<D> du;
  
  Mask<D> o;
  Vec<D> u, s, t, rx, ry;
  Vec2<D> r;

  u = Mul(d, Set(df, 4.0));
  Vec<RebindToSigned<D>> q = ConvertTo(di, Trunc(u));
  q = And(Add(q, Xor(BitCast(di, ShiftRight<63>(BitCast(du, q))), Set(di, 1))), Set(di, ~1));
  s = Sub(u, ConvertTo(df, q));

  t = s;
  s = Mul(s, s);
  
  

  u = Set(df, +0.6880638894766060136e-11);
  u = MulAdd(u, s, Set(df, -0.1757159564542310199e-8));
  u = MulAdd(u, s, Set(df, +0.3133616327257867311e-6));
  u = MulAdd(u, s, Set(df, -0.3657620416388486452e-4));
  u = MulAdd(u, s, Set(df, +0.2490394570189932103e-2));
  u = MulAdd(u, s, Set(df, -0.8074551218828056320e-1));
  u = MulAdd(u, s, Set(df, +0.7853981633974482790e+0));

  rx = Mul(u, t);

  
  
  u = Set(df, -0.3860141213683794352e-12);
  u = MulAdd(u, s, Set(df, +0.1150057888029681415e-9));
  u = MulAdd(u, s, Set(df, -0.2461136493006663553e-7));
  u = MulAdd(u, s, Set(df, +0.3590860446623516713e-5));
  u = MulAdd(u, s, Set(df, -0.3259918869269435942e-3));
  u = MulAdd(u, s, Set(df, +0.1585434424381541169e-1));
  u = MulAdd(u, s, Set(df, -0.3084251375340424373e+0));
  u = MulAdd(u, s, Set(df, 1));

  ry = u;

  

  o = RebindMask(df, Eq(And(q, Set(di, 2)), Set(di, 0)));
  r = Create2(df, IfThenElse(o, rx, ry), IfThenElse(o, ry, rx));

  o = RebindMask(df, Eq(And(q, Set(di, 4)), Set(di, 4)));
  r = Set2<0>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<0>(r)))));

  o = RebindMask(df, Eq(And(Add(q, Set(di, 2)), Set(di, 4)), Set(di, 4)));
  r = Set2<1>(r, BitCast(df, Xor(IfThenElseZero(RebindMask(du, o), BitCast(du, Set(df, -0.0))), BitCast(du, Get2<1>(r)))));

  o = Gt(Abs(d), Set(df, TrigRangeMax3/4));
  r = Set2<0>(r, BitCast(df, IfThenZeroElse(RebindMask(du, o), BitCast(du, Get2<0>(r)))));
  r = Set2<1>(r, BitCast(df, IfThenZeroElse(RebindMask(du, o), BitCast(du, Get2<1>(r)))));

  o = IsInf(d);
  r = Set2<0>(r, BitCast(df, IfThenElse(RebindMask(du, o), Set(du, -1), BitCast(du, Get2<0>(r)))));
  r = Set2<1>(r, BitCast(df, IfThenElse(RebindMask(du, o), Set(du, -1), BitCast(du, Get2<1>(r)))));

  return r;
}

// Computes sin(x*pi) with max(0.506 ULP, DBL_MIN) accuracy (in practice staying below 2 ULP even for subnormals)
// Translated from libm/sleefsimddp.c:1456 xsinpi_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) SinPi(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  
  Vec2<D> x = SinPiDD(df, d);
  Vec<D> r = Add(Get2<0>(x), Get2<1>(x));

  r = IfThenElse(Eq(d, Set(df, -0.0)), Set(df, -0.0), r);
  r = BitCast(df, IfThenZeroElse(RebindMask(du, Gt(Abs(d), Set(df, TrigRangeMax3/4))), BitCast(du, r)));
  r = BitCast(df, IfThenElse(RebindMask(du, IsInf(d)), Set(du, -1), BitCast(du, r)));
  
  return r;
}

// Computes cos(x*pi) with max(0.506 ULP, DBL_MIN) accuracy (in practice staying below 2 ULP even for subnormals)
// Translated from libm/sleefsimddp.c:1507 xcospi_u05
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) CosPi(const D df, Vec<D> d) {
  RebindToUnsigned<D> du;
  
  Vec2<D> x = CosPiDD(df, d);
  Vec<D> r = Add(Get2<0>(x), Get2<1>(x));

  r = IfThenElse(Gt(Abs(d), Set(df, TrigRangeMax3/4)), Set(df, 1), r);
  r = BitCast(df, IfThenElse(RebindMask(du, IsInf(d)), Set(du, -1), BitCast(du, r)));
  
  return r;
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
  
  Vec<D> e = Expm1Fast(df, Abs(x));

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
  
  Vec<D> d = Expm1Fast(df, Mul(Set(df, 2), Abs(x)));
  Vec<D> y = Div(d, Add(Set(df, 2), d));

  y = IfThenElse(Or(Gt(Abs(x), Set(df, 18.714973875)), IsNaN(y)), Set(df, 1.0), y);
  y = MulSignBit(df, y, x);
  y = BitCast(df, IfThenElse(RebindMask(du, IsNaN(x)), Set(du, -1), BitCast(du, y)));

  return y;
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

// Computes erf(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:3470 xerf_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Erf(const D df, Vec<D> a) {
  Vec<D> t, x = Abs(a);
  Vec2<D> t2;
  Vec<D> x2 = Mul(x, x), x4 = Mul(x2, x2);
  Vec<D> x8 = Mul(x4, x4), x16 = Mul(x8, x8);
  Mask<D> o25 = Le(x, Set(df, 2.5));

  if (HWY_LIKELY(AllTrue(df, o25))) {
    // Abramowitz and Stegun
    t = Estrin(x, x2, x4, x8, x16, Set(df, +0.1490149719145544729e-3), Set(df, +0.2323253155213076174e-3), Set(df, +0.1060862922597579532e-3), Set(df, -0.3309403072749947546e-4), Set(df, +0.3860236356493129101e-5), Set(df, +0.3417987836115362136e-5), Set(df, -0.1842918273003998283e-5), Set(df, +0.2945514529987331866e-6), Set(df, +0.1389000557865837204e-6), Set(df, -0.9808074602255194288e-7), Set(df, +0.1669878756181250355e-7), Set(df, +0.1215442362680889243e-7), Set(df, -0.1143939895758628484e-7), Set(df, +0.5435081826716212389e-8), Set(df, -0.1808044474288848915e-8), Set(df, +0.4507647462598841629e-9), Set(df, -0.8499973178354613440e-10), Set(df, +0.1186474230821585259e-10), Set(df, -0.1162238220110999364e-11), Set(df, +0.7151909970790897009e-13), Set(df, -0.2083271002525222097e-14));
    t2 = Poly4DD(df, x, t,
		 Create2(df, Set(df, 0.0092877958392275604405), Set(df, 7.9287559463961107493e-19)),
		 Create2(df, Set(df, 0.042275531758784692937), Set(df, 1.3785226620501016138e-19)),
		 Create2(df, Set(df, 0.07052369794346953491), Set(df, 9.5846628070792092842e-19)));
    t2 = AddFastDD(df, Set(df, 1), MulDD(df, t2, x));
    t2 = SquareDD(df, t2);
    t2 = SquareDD(df, t2);
    t2 = SquareDD(df, t2);
    t2 = SquareDD(df, t2);
    t2 = RecDD(df, t2);
  } else {
    t = Estrin(x, x2, x4, x8, x16, IfThenElse(o25, Set(df, +0.1490149719145544729e-3), Set(df, -0.1021557155453465954e+0)), IfThenElse(o25, Set(df, +0.2323253155213076174e-3), Set(df, +0.1820191395263313222e-1)), IfThenElse(o25, Set(df, +0.1060862922597579532e-3), Set(df, +0.1252823202436093193e-2)), IfThenElse(o25, Set(df, -0.3309403072749947546e-4), Set(df, -0.2605566912579998680e-2)), IfThenElse(o25, Set(df, +0.3860236356493129101e-5), Set(df, +0.1210736097958368864e-2)), IfThenElse(o25, Set(df, +0.3417987836115362136e-5), Set(df, -0.3551065097428388658e-3)), IfThenElse(o25, Set(df, -0.1842918273003998283e-5), Set(df, +0.6539731269664907554e-4)), IfThenElse(o25, Set(df, +0.2945514529987331866e-6), Set(df, -0.2515023395879724513e-5)), IfThenElse(o25, Set(df, +0.1389000557865837204e-6), Set(df, -0.3341634127317201697e-5)), IfThenElse(o25, Set(df, -0.9808074602255194288e-7), Set(df, +0.1517272660008588485e-5)), IfThenElse(o25, Set(df, +0.1669878756181250355e-7), Set(df, -0.4160448058101303405e-6)), IfThenElse(o25, Set(df, +0.1215442362680889243e-7), Set(df, +0.8525705726469103499e-7)), IfThenElse(o25, Set(df, -0.1143939895758628484e-7), Set(df, -0.1380745342355033142e-7)), IfThenElse(o25, Set(df, +0.5435081826716212389e-8), Set(df, +0.1798167853032159309e-8)), IfThenElse(o25, Set(df, -0.1808044474288848915e-8), Set(df, -0.1884658558040203709e-9)), IfThenElse(o25, Set(df, +0.4507647462598841629e-9), Set(df, +0.1573695559331945583e-10)), IfThenElse(o25, Set(df, -0.8499973178354613440e-10), Set(df, -0.1025221466851463164e-11)), IfThenElse(o25, Set(df, +0.1186474230821585259e-10), Set(df, +0.5029618322872872715e-13)), IfThenElse(o25, Set(df, -0.1162238220110999364e-11), Set(df, -0.1749316241455644088e-14)), IfThenElse(o25, Set(df, +0.7151909970790897009e-13), Set(df, +0.3847193332817048172e-16)), IfThenElse(o25, Set(df, -0.2083271002525222097e-14), Set(df, -0.4024015130752621932e-18)));
    t2 = Poly4DD(df, x, t,
		 IfThenElse(df, o25, Create2(df, Set(df, 0.0092877958392275604405), Set(df, 7.9287559463961107493e-19)),
				     Create2(df, Set(df, -0.63691044383641748361), Set(df, -2.4249477526539431839e-17))),
		 IfThenElse(df, o25, Create2(df, Set(df, 0.042275531758784692937), Set(df, 1.3785226620501016138e-19)),
				     Create2(df, Set(df, -1.1282926061803961737), Set(df, -6.2970338860410996505e-17))),
		 IfThenElse(df, o25, Create2(df, Set(df, 0.07052369794346953491), Set(df, 9.5846628070792092842e-19)),
				     Create2(df, Set(df, -1.2261313785184804967e-05), Set(df, -5.5329707514490107044e-22))));
    Vec2<D> s2 = AddFastDD(df, Set(df, 1), MulDD(df, t2, x));
    s2 = SquareDD(df, s2);
    s2 = SquareDD(df, s2);
    s2 = SquareDD(df, s2);
    s2 = SquareDD(df, s2);
    s2 = RecDD(df, s2);
    t2 = IfThenElse(df, o25, s2, Create2(df, ExpDD_double(df, t2), Set(df, 0)));
  }

  t2 = AddDD(df, t2, Set(df, -1));

  Vec<D> z = Neg(Add(Get2<0>(t2), Get2<1>(t2)));
  z = IfThenElse(Lt(x, Set(df, 1e-8)), Mul(x, Set(df, 1.12837916709551262756245475959)), z);
  z = IfThenElse(Ge(x, Set(df, 6)), Set(df, 1), z);
  z = IfThenElse(IsInf(a), Set(df, 1), z);
  z = IfThenElse(Eq(a, Set(df, 0)), Set(df, 0), z);
  z = MulSignBit(df, z, a);

  return z;
}

// Computes 1 - erf(x) with 1.5 ULP accuracy
// Translated from libm/sleefsimddp.c:3565 xerfc_u15
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Erfc(const D df, Vec<D> a) {
  Vec<D> s = a, r = Set(df, 0), t;
  Vec2<D> u, d, x;
  a = Abs(a);
  Mask<D> o0 = Lt(a, Set(df, 1.0));
  Mask<D> o1 = Lt(a, Set(df, 2.2));
  Mask<D> o2 = Lt(a, Set(df, 4.2));
  Mask<D> o3 = Lt(a, Set(df, 27.3));

  u = IfThenElse(df, o0, MulDD(df, a, a), IfThenElse(df, o1, Create2(df, a, Set(df, 0)), DivDD(df, Create2(df, Set(df, 1), Set(df, 0)), Create2(df, a, Set(df, 0)))));

  t = IfThenElse(o0, Set(df, +0.6801072401395386139e-20), IfThenElse(o1, Set(df, +0.3438010341362585303e-12), IfThenElse(o2, Set(df, -0.5757819536420710449e+2), Set(df, +0.2334249729638701319e+5))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, -0.2161766247570055669e-18), IfThenElse(o1, Set(df, -0.1237021188160598264e-10), IfThenElse(o2, Set(df, +0.4669289654498104483e+3), Set(df, -0.4695661044933107769e+5)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, +0.4695919173301595670e-17), IfThenElse(o1, Set(df, +0.2117985839877627852e-09), IfThenElse(o2, Set(df, -0.1796329879461355858e+4), Set(df, +0.3173403108748643353e+5)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, -0.9049140419888007122e-16), IfThenElse(o1, Set(df, -0.2290560929177369506e-08), IfThenElse(o2, Set(df, +0.4355892193699575728e+4), Set(df, +0.3242982786959573787e+4)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, +0.1634018903557410728e-14), IfThenElse(o1, Set(df, +0.1748931621698149538e-07), IfThenElse(o2, Set(df, -0.7456258884965764992e+4), Set(df, -0.2014717999760347811e+5)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, -0.2783485786333451745e-13), IfThenElse(o1, Set(df, -0.9956602606623249195e-07), IfThenElse(o2, Set(df, +0.9553977358167021521e+4), Set(df, +0.1554006970967118286e+5)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, +0.4463221276786415752e-12), IfThenElse(o1, Set(df, +0.4330010240640327080e-06), IfThenElse(o2, Set(df, -0.9470019905444229153e+4), Set(df, -0.6150874190563554293e+4)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, -0.6711366622850136563e-11), IfThenElse(o1, Set(df, -0.1435050600991763331e-05), IfThenElse(o2, Set(df, +0.7387344321849855078e+4), Set(df, +0.1240047765634815732e+4)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, +0.9422759050232662223e-10), IfThenElse(o1, Set(df, +0.3460139479650695662e-05), IfThenElse(o2, Set(df, -0.4557713054166382790e+4), Set(df, -0.8210325475752699731e+2)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, -0.1229055530100229098e-08), IfThenElse(o1, Set(df, -0.4988908180632898173e-05), IfThenElse(o2, Set(df, +0.2207866967354055305e+4), Set(df, +0.3242443880839930870e+2)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, +0.1480719281585086512e-07), IfThenElse(o1, Set(df, -0.1308775976326352012e-05), IfThenElse(o2, Set(df, -0.8217975658621754746e+3), Set(df, -0.2923418863833160586e+2)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, -0.1636584469123399803e-06), IfThenElse(o1, Set(df, +0.2825086540850310103e-04), IfThenElse(o2, Set(df, +0.2268659483507917400e+3), Set(df, +0.3457461732814383071e+0)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, +0.1646211436588923575e-05), IfThenElse(o1, Set(df, -0.6393913713069986071e-04), IfThenElse(o2, Set(df, -0.4633361260318560682e+2), Set(df, +0.5489730155952392998e+1)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, -0.1492565035840623511e-04), IfThenElse(o1, Set(df, -0.2566436514695078926e-04), IfThenElse(o2, Set(df, +0.9557380123733945965e+1), Set(df, +0.1559934132251294134e-2)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, +0.1205533298178967851e-03), IfThenElse(o1, Set(df, +0.5895792375659440364e-03), IfThenElse(o2, Set(df, -0.2958429331939661289e+1), Set(df, -0.1541741566831520638e+1)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, -0.8548327023450850081e-03), IfThenElse(o1, Set(df, -0.1695715579163588598e-02), IfThenElse(o2, Set(df, +0.1670329508092765480e+0), Set(df, +0.2823152230558364186e-5)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, +0.5223977625442187932e-02), IfThenElse(o1, Set(df, +0.2089116434918055149e-03), IfThenElse(o2, Set(df, +0.6096615680115419211e+0), Set(df, +0.6249999184195342838e+0)))));
  t = MulAdd(t, Get2<0>(u), IfThenElse(o0, Set(df, -0.2686617064513125222e-01), IfThenElse(o1, Set(df, +0.1912855949584917753e-01), IfThenElse(o2, Set(df, +0.1059212443193543585e-2), Set(df, +0.1741749416408701288e-8)))));

  d = MulDD(df, u, t);
  d = AddDD(df, d, Create2(df, IfThenElse(o0, Set(df, 0.11283791670955126141), IfThenElse(o1, Set(df, -0.10277263343147646779), IfThenElse(o2, Set(df, -0.50005180473999022439), Set(df, -0.5000000000258444377)))), IfThenElse(o0, Set(df, -4.0175691625932118483e-18), IfThenElse(o1, Set(df, -6.2338714083404900225e-18), IfThenElse(o2, Set(df, 2.6362140569041995803e-17), Set(df, -4.0074044712386992281e-17))))));
  d = MulDD(df, d, u);
  d = AddDD(df, d, Create2(df, IfThenElse(o0, Set(df, -0.37612638903183753802), IfThenElse(o1, Set(df, -0.63661976742916359662), IfThenElse(o2, Set(df, 1.601106273924963368e-06), Set(df, 2.3761973137523364792e-13)))), IfThenElse(o0, Set(df, 1.3391897206042552387e-17), IfThenElse(o1, Set(df, 7.6321019159085724662e-18), IfThenElse(o2, Set(df, 1.1974001857764476775e-23), Set(df, -1.1670076950531026582e-29))))));
  d = MulDD(df, d, u);
  d = AddDD(df, d, Create2(df, IfThenElse(o0, Set(df, 1.1283791670955125586), IfThenElse(o1, Set(df, -1.1283791674717296161), IfThenElse(o2, Set(df, -0.57236496645145429341), Set(df, -0.57236494292470108114)))), IfThenElse(o0, Set(df, 1.5335459613165822674e-17), IfThenElse(o1, Set(df, 8.0896847755965377194e-17), IfThenElse(o2, Set(df, 3.0704553245872027258e-17), Set(df, -2.3984352208056898003e-17))))));
  
  x = MulDD(df, IfThenElse(df, o1, d, Create2(df, Neg(a), Set(df, 0))), a);
  x = IfThenElse(df, o1, x, AddDD(df, x, d));
  x = IfThenElse(df, o0, SubDD(df, Create2(df, Set(df, 1), Set(df, 0)), x), ExpDD(df, x));
  x = IfThenElse(df, o1, x, MulDD(df, x, u));

  r = IfThenElse(o3, Add(Get2<0>(x), Get2<1>(x)), Set(df, 0));
  r = IfThenElse(SignBitMask(df, s), Sub(Set(df, 2), r), r);
  r = IfThenElse(IsNaN(s), Set(df, NanDouble), r);
  return r;
}

// Computes gamma(x) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:3427 xtgamma_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Gamma(const D df, Vec<D> a) {
  Vec4<D> d = GammaQD(df, a);
  Vec2<D> y = MulDD(df, ExpDD(df, Create2(df, Get4<0>(d), Get4<1>(d))), Create2(df, Get4<2>(d), Get4<3>(d)));
  Vec<D> r = Add(Get2<0>(y), Get2<1>(y));
  Mask<D> o;

  o = Or(Or(Eq(a, Set(df, -InfDouble)), And(Lt(a, Set(df, 0)), IsInt(df, a))), And(And(IsFinite(a), Lt(a, Set(df, 0))), IsNaN(r)));
  r = IfThenElse(o, Set(df, NanDouble), r);

  o = And(And(Or(Eq(a, Set(df, InfDouble)), IsFinite(a)), Ge(a, Set(df, -DoubleMin))), Or(Or(Eq(a, Set(df, 0)), Gt(a, Set(df, 200))), IsNaN(r)));
  r = IfThenElse(o, MulSignBit(df, Set(df, InfDouble), a), r);
  
  return r;
}

// Computes log(gamma(x)) with 1.0 ULP accuracy
// Translated from libm/sleefsimddp.c:3446 xlgamma_u1
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) LogGamma(const D df, Vec<D> a) {
  Vec4<D> d = GammaQD(df, a);
  Vec2<D> y = AddDD(df, Create2(df, Get4<0>(d), Get4<1>(d)), LogFastDD(df, AbsDD(df, Create2(df, Get4<2>(d), Get4<3>(d)))));
  Vec<D> r = Add(Get2<0>(y), Get2<1>(y));
  Mask<D> o;

  o = Or(IsInf(a), Or(And(Le(a, Set(df, 0)), IsInt(df, a)), And(IsFinite(a), IsNaN(r))));
  r = IfThenElse(o, Set(df, InfDouble), r);

  return r;
}

// Computes fmod(x), the floating point remainder
// Translated from libm/sleefsimddp.c:3269 xfmod
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Fmod(const D df, Vec<D> x, Vec<D> y) {
  RebindToUnsigned<D> du;
  
  Vec<D> n = Abs(x), d = Abs(y), s = Set(df, 1), q;
  Mask<D> o = Lt(d, Set(df, DoubleMin));
  n = IfThenElse(o, Mul(n, Set(df, UINT64_C(1) << 54)), n);
  d = IfThenElse(o, Mul(d, Set(df, UINT64_C(1) << 54)), d);
  s  = IfThenElse(o, Mul(s, Set(df, 1.0 / (UINT64_C(1) << 54))), s);
  Vec2<D> r = Create2(df, n, Set(df, 0));
  Vec<D> rd = Toward0(df, Div(Set(df, 1.0), d));

  for(int i=0;i<21;i++) { // ceil(log2(DBL_MAX) / 52)
    q = Trunc(Mul(Toward0(df, Get2<0>(r)), rd));
#ifndef HWY_SLEEF_HAS_FMA
    q = BitCast(df, And(BitCast(du, q), Set(du, UINT64_C(0xfffffffffffffffe))));
#endif
    q = IfThenElse(And(Gt(Mul(Set(df, 3), d), Get2<0>(r)), Ge(Get2<0>(r), d)), Set(df, 2), q);
    q = IfThenElse(And(Gt(Add(d, d), Get2<0>(r)), Ge(Get2<0>(r), d)), Set(df, 1), q);
    r = NormalizeDD(df, AddDD(df, r, MulDD(df, q, Neg(d))));
    if (AllTrue(df, Lt(Get2<0>(r), d))) break;
  }
  
  Vec<D> ret = Mul(Get2<0>(r), s);
  ret = IfThenElse(Eq(Add(Get2<0>(r), Get2<1>(r)), d), Set(df, 0), ret);

  ret = MulSignBit(df, ret, x);

  ret = IfThenElse(Lt(n, d), x, ret);
  ret = IfThenElse(Eq(d, Set(df, 0)), Set(df, NanDouble), ret);

  return ret;
}

// Computes remainder(x), the signed floating point remainder
// Translated from libm/sleefsimddp.c:3314 xremainder
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) Remainder(const D df, Vec<D> x, Vec<D> y) {
  RebindToUnsigned<D> du;
  
  Vec<D> n = Abs(x), d = Abs(y), s = Set(df, 1), q;
  Mask<D> o = Lt(d, Set(df, DoubleMin*2));
  n = IfThenElse(o, Mul(n, Set(df, UINT64_C(1) << 54)), n);
  d = IfThenElse(o, Mul(d, Set(df, UINT64_C(1) << 54)), d);
  s  = IfThenElse(o, Mul(s, Set(df, 1.0 / (UINT64_C(1) << 54))), s);
  Vec<D> rd = Div(Set(df, 1.0), d);
  Vec2<D> r = Create2(df, n, Set(df, 0));
  Mask<D> qisodd = Ne(Set(df, 0), Set(df, 0));

  for(int i=0;i<21;i++) { // ceil(log2(DBL_MAX) / 52)
    q = Round(Mul(Get2<0>(r), rd));
#ifndef HWY_SLEEF_HAS_FMA
    q = BitCast(df, And(BitCast(du, q), Set(du, UINT64_C(0xfffffffffffffffe))));
#endif
    q = IfThenElse(Lt(Abs(Get2<0>(r)), Mul(d, Set(df, 1.5))), MulSignBit(df, Set(df, 1.0), Get2<0>(r)), q);
    q = IfThenElse(Or(Lt(Abs(Get2<0>(r)), Mul(d, Set(df, 0.5))), AndNot(qisodd, Eq(Abs(Get2<0>(r)), Mul(d, Set(df, 0.5))))), Set(df, 0.0), q);
    if (AllTrue(df, Eq(q, Set(df, 0)))) break;
    q = IfThenElse(IsInf(Mul(q, Neg(d))), Add(q, MulSignBit(df, Set(df, -1), Get2<0>(r))), q);
    qisodd = Xor(qisodd, IsOdd(df, q));
    r = NormalizeDD(df, AddDD(df, r, MulDD(df, q, Neg(d))));
  }
  
  Vec<D> ret = Mul(Get2<0>(r), s);
  ret = MulSignBit(df, ret, x);
  ret = IfThenElse(IsInf(y), IfThenElse(IsInf(x), Set(df, NanDouble), x), ret);
  ret = IfThenElse(Eq(d, Set(df, 0)), Set(df, NanDouble), ret);
  return ret;
}

// Computes x * 2^exp
// Translated from libm/sleefsimddp.c:338 xldexp
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) LdExp(const D df, Vec<D> x, Vec<RebindToSigned<D>> q) {
 return LoadExp(df, x, q); }

// Decomposes x into 2^exp * fr where abs(fr) is in [0.5, 1), returning fr
// Translated from libm/sleefsimddp.c:3079 xfrfrexp
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) FrFrexp(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  
  x = IfThenElse(Lt(Abs(x), Set(df, DoubleMin)), Mul(x, Set(df, UINT64_C(1) << 63)), x);

  Vec<RebindToUnsigned<D>> xm = BitCast(du, x);
  xm = And(xm, Set(du, ~INT64_C(0x7ff0000000000000)));
  xm = Or(xm, Set(du, INT64_C(0x3fe0000000000000)));

  Vec<D> ret = BitCast(df, xm);

  ret = IfThenElse(IsInf(x), MulSignBit(df, Set(df, InfDouble), x), ret);
  ret = IfThenElse(Eq(x, Set(df, 0)), x, ret);
  
  return ret;
}

// Decomposes x into 2^exp * fr where abs(fr) is in [0.5, 1), returning exp
// Translated from libm/sleefsimddp.c:3094 xexpfrexp
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<RebindToSigned<D>>) ExpFrexp(const D df, Vec<D> x) {
  RebindToUnsigned<D> du;
  RebindToSigned<D> di;
  
  x = IfThenElse(Lt(Abs(x), Set(df, DoubleMin)), Mul(x, Set(df, UINT64_C(1) << 63)), x);

  Vec<RebindToSigned<D>> ret = BitCast(di, ShiftRight<32>(BitCast(du, x)));
  ret = Sub(And(BitCast(di, ShiftRight<20>(BitCast(du, ret))), Set(di, 0x7ff)), Set(di, 0x3fe));

  ret = IfThenElse(RebindMask(di, Or(Or(Eq(x, Set(df, 0)), IsNaN(x)), IsInf(x))), Set(di, 0), ret);
  
  return ret;
}

// Computes the unbiased exponent of x
// Translated from libm/sleefsimddp.c:340 xilogb
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<RebindToSigned<D>>) ILogB(const D df, Vec<D> d) {
  RebindToSigned<D> di;
  
  Vec<D> e = ConvertTo(df, ILogB1(df, Abs(d)));
  e = IfThenElse(Eq(d, Set(df, 0)), Set(df, ILogB0), e);
  e = IfThenElse(IsNaN(d), Set(df, ILogBNan), e);
  e = IfThenElse(IsInf(d), Set(df, IntMax), e);
  return ConvertTo(di, Round(e));
}

// Decompose x into an integer and fractional part
// Translated from libm/sleefsimddp.c:1371 XMODF
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec2<D>) Modf(const D df, Vec<D> x) {
  RebindToSigned<D> di;
  
  Vec<D> fr = Sub(x, Mul(Set(df, INT64_C(1) << 31), ConvertTo(df, ConvertTo(di, Trunc(Mul(x, Set(df, 1.0 / (INT64_C(1) << 31))))))));
  fr = Sub(fr, ConvertTo(df, ConvertTo(di, Trunc(fr))));
  fr = IfThenElse(Gt(Abs(x), Set(df, INT64_C(1) << 52)), Set(df, 0), fr);

  Vec2<D> ret;

  ret = Create2(df, CopySign(df, fr, x), CopySign(df, Sub(x, fr), x));

  return ret;
}

// Returns the next representable value after x in the direction of y
// Translated from libm/sleefsimddp.c:3056 xnextafter
template<class D>
HWY_INLINE HWY_SLEEF_IF_DOUBLE(D, Vec<D>) NextAfter(const D df, Vec<D> x, Vec<D> y) {
  RebindToUnsigned<D> du;
  
  x = IfThenElse(Eq(x, Set(df, 0)), MulSignBit(df, Set(df, 0), y), x);
  Vec<RebindToUnsigned<D>> xi2 = BitCast(du, x);
  Mask<D> c = Xor(SignBitMask(df, x), Ge(y, x));

  xi2 = IfThenElse(RebindMask(du, c), Neg(Xor(xi2, Set(du, (static_cast<uint64_t>((int)(1U << 31)) << 32) | 0))), xi2);

  xi2 = IfThenElse(RebindMask(du, Ne(x, y)), Sub(xi2, Set(du, (static_cast<uint64_t>(0) << 32) | 1)), xi2);

  xi2 = IfThenElse(RebindMask(du, c), Neg(Xor(xi2, Set(du, (static_cast<uint64_t>((int)(1U << 31)) << 32) | 0))), xi2);

  Vec<D> ret = BitCast(df, xi2);

  ret = IfThenElse(And(Eq(ret, Set(df, 0)), Ne(x, Set(df, 0))), MulSignBit(df, Set(df, 0), x), ret);

  ret = IfThenElse(And(Eq(x, Set(df, 0)), Eq(y, Set(df, 0))), y, ret);

  ret = IfThenElse(Or(IsNaN(x), IsNaN(y)), Set(df, NanDouble), ret);
  
  return ret;
}

}  // namespace sleef
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif  // HIGHWAY_HWY_CONTRIB_SLEEF_SLEEF_INL_

