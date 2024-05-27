#ifndef DEFINE_DISPATCHERS_H_
#define DEFINE_DISPATCHERS_H_

#include "hwy/contrib/math/math-inl.h"
#include "sleef.h"
#include "src/gen-bindings/sleef-generated.h"

// Define functions with signature f(const float *in, size_t n, float
// *__restrict__ out) These all are defined just for AVX2 Each op has a *Sleef,
// *Hwy, and *Translated variant

// Register an AVX2 highway wrapper for a Sleef function
#define SLEEF_TO_HWY(NAME, FNf8, FNd4)                   \
  HWY_BEFORE_NAMESPACE();                                \
  namespace hwy {                                        \
  namespace N_AVX2 {                                     \
  template <class D>                                     \
  HWY_INLINE Vec<D> NAME(const D df, Vec256<float> d) {  \
    return Vec256<float>{FNf8(d.raw)};                   \
  }                                                      \
  template <class D>                                     \
  HWY_INLINE Vec<D> NAME(const D df, Vec256<double> d) { \
    return Vec256<double>{FNd4(d.raw)};                   \
  }                                                      \
  }                                                      \
  }                                                      \
  HWY_AFTER_NAMESPACE();

#define WRAP_OP1(NAME, FN)                                \
  HWY_BEFORE_NAMESPACE();                                 \
  namespace hwy {                                         \
  namespace HWY_NAMESPACE {                               \
  template <typename T>                                   \
  void NAME(const T *in, size_t n, T *__restrict__ out) { \
    using D = ScalableTag<T>;                             \
    D d;                                                  \
    Vec<D> v;                                             \
    size_t lanes = Lanes(d);                              \
    for (size_t i = 0; i < n; i += lanes) {               \
      v = Load(d, in + i);                                \
      v = FN(d, v);                                       \
      Store(v, d, out + i);                               \
    }                                                     \
  }                                                       \
  }                                                       \
  }                                                       \
  HWY_AFTER_NAMESPACE();

#define WRAP_SCALAR1(NAME, FN)                            \
  template <typename T>                                   \
  void NAME(const T *in, size_t n, T *__restrict__ out) { \
    for (size_t i = 0; i < n; i += 1) {                   \
      out[i] = FN(in[i]);                                 \
    }                                                     \
  }

#define DISPATCH_OP1(NAME, FN)                                         \
  namespace hwy {                                                      \
  template <typename T>                                                \
  HWY_NOINLINE void NAME(const T *in, size_t n, T *__restrict__ out) { \
    HWY_STATIC_DISPATCH(NAME)(in, n, out);                             \
  }                                                                    \
  }

#define DISPATCH_AND_WRAP_OP1(NAME, FN) \
  WRAP_OP1(NAME, FN)                    \
  DISPATCH_OP1(NAME, FN)

// Define dispatch names based on common naming expectations for operations
#define DISPATCH_ALL(OP, STDFN, SLEEF_F32, SLEEF_F64) \
  DISPATCH_AND_WRAP_OP1(OP##Hwy, OP)                  \
  DISPATCH_AND_WRAP_OP1(OP##Translated, sleef::OP)    \
  SLEEF_TO_HWY(OP##Sleef, SLEEF_F32, SLEEF_F64)       \
  DISPATCH_AND_WRAP_OP1(OP##Sleef, OP##Sleef)         \
  WRAP_SCALAR1(OP##Std, STDFN)

#define DISPATCH_SLEEF_ONLY(OP, SLEEF_F32, SLEEF_F64) \
  DISPATCH_AND_WRAP_OP1(OP##Translated, sleef::OP)    \
  SLEEF_TO_HWY(OP##Sleef, SLEEF_F32, SLEEF_F64)       \
  DISPATCH_AND_WRAP_OP1(OP##Sleef, OP##Sleef)

#define DISPATCH_ALL_SKIP_HWY(OP, STDFN, SLEEF_F32, SLEEF_F64) \
  DISPATCH_AND_WRAP_OP1(OP##Translated, sleef::OP)             \
  SLEEF_TO_HWY(OP##Sleef, SLEEF_F32, SLEEF_F64)                \
  DISPATCH_AND_WRAP_OP1(OP##Sleef, OP##Sleef)                  \
  WRAP_SCALAR1(OP##Std, STDFN)

// Define dispatch names for low-precision variants which also have a high
// precision variant
#define DISPATCH_ALL_LOW_PRECISION(OP, SLEEF_F32, SLEEF_F64) \
  DISPATCH_AND_WRAP_OP1(OP##FastTranslated, sleef::OP##Fast) \
  SLEEF_TO_HWY(OP##FastSleef, SLEEF_F32, SLEEF_F64)          \
  DISPATCH_AND_WRAP_OP1(OP##FastSleef, OP##FastSleef)

template <typename T>
T exp10_helper(T x);

template <>
float exp10_helper(float x) {
  return exp10f(x);
}

template <>
double exp10_helper(double x) {
  return exp10(x);
}

// clang-format off
DISPATCH_ALL(Exp, std::exp, Sleef_finz_expf8_u10avx2, Sleef_finz_expd4_u10avx2)
DISPATCH_ALL(Exp2, std::exp2, Sleef_finz_exp2f8_u10avx2, Sleef_finz_exp2d4_u10avx2)
DISPATCH_ALL(Exp10, exp10_helper, Sleef_finz_exp10f8_u10avx2, Sleef_finz_exp10d4_u10avx2)
DISPATCH_ALL(Expm1, std::expm1, Sleef_finz_expm1f8_u10avx2, Sleef_finz_expm1d4_u10avx2)
DISPATCH_ALL(Log, std::log, Sleef_finz_logf8_u10avx2, Sleef_finz_logd4_u10avx2)
DISPATCH_ALL_LOW_PRECISION(Log, Sleef_finz_logf8_u35avx2, Sleef_finz_logd4_u35avx2)
DISPATCH_ALL(Log2, std::log2, Sleef_finz_log2f8_u10avx2, Sleef_finz_log2d4_u10avx2)
DISPATCH_ALL(Log10, std::log10, Sleef_finz_log10f8_u10avx2, Sleef_finz_log10d4_u10avx2)
DISPATCH_ALL(Log1p, std::log1p, Sleef_finz_log1pf8_u10avx2, Sleef_finz_log1pd4_u10avx2)
// Skip pow
DISPATCH_ALL_SKIP_HWY(Sqrt, std::sqrt, Sleef_finz_sqrtf8_avx2, Sleef_finz_sqrtd4_u05avx2)
DISPATCH_ALL_LOW_PRECISION(Sqrt, Sleef_finz_sqrtf8_u35avx2, Sleef_finz_sqrtd4_u35avx2)
DISPATCH_ALL_SKIP_HWY(Cbrt, std::cbrt, Sleef_finz_cbrtf8_u10avx2, Sleef_finz_cbrtd4_u10avx2)
DISPATCH_ALL_LOW_PRECISION(Cbrt, Sleef_finz_cbrtf8_u35avx2, Sleef_finz_cbrtd4_u35avx2)
// Skip hypot

DISPATCH_ALL(Sin, std::sin, Sleef_finz_sinf8_u10avx2, Sleef_finz_sind4_u10avx2)
DISPATCH_ALL_LOW_PRECISION(Sin, Sleef_finz_sinf8_u35avx2, Sleef_finz_sind4_u35avx2)
DISPATCH_ALL(Cos, std::cos, Sleef_finz_cosf8_u10avx2, Sleef_finz_cosd4_u10avx2)
DISPATCH_ALL_LOW_PRECISION(Cos, Sleef_finz_cosf8_u35avx2, Sleef_finz_cosd4_u35avx2)
DISPATCH_ALL_SKIP_HWY(Tan, std::tan, Sleef_finz_tanf8_u10avx2, Sleef_finz_tand4_u10avx2)
DISPATCH_ALL_LOW_PRECISION(Tan, Sleef_finz_tanf8_u35avx2, Sleef_finz_tand4_u35avx2)

DISPATCH_ALL(Asin, std::asin, Sleef_finz_asinf8_u10avx2, Sleef_finz_asind4_u10avx2)
DISPATCH_ALL_LOW_PRECISION(Asin, Sleef_finz_asinf8_u35avx2, Sleef_finz_asind4_u35avx2)
DISPATCH_ALL(Acos, std::acos, Sleef_finz_acosf8_u10avx2, Sleef_finz_acosd4_u10avx2)
DISPATCH_ALL_LOW_PRECISION(Acos, Sleef_finz_acosf8_u35avx2, Sleef_finz_acosd4_u35avx2)
DISPATCH_ALL(Atan, std::atan, Sleef_finz_atanf8_u10avx2, Sleef_finz_atand4_u10avx2)
DISPATCH_ALL_LOW_PRECISION(Atan, Sleef_finz_atanf8_u35avx2, Sleef_finz_atand4_u35avx2)
// Skip Atan2

DISPATCH_SLEEF_ONLY(SinPi, Sleef_finz_sinpif8_u05avx2, Sleef_finz_sinpid4_u05avx2)
DISPATCH_SLEEF_ONLY(CosPi, Sleef_finz_cospif8_u05avx2, Sleef_finz_cospid4_u05avx2)
// Skip SinCos

DISPATCH_ALL(Sinh, std::sinh, Sleef_finz_sinhf8_u10avx2, Sleef_finz_sinhd4_u10avx2)
DISPATCH_ALL_LOW_PRECISION(Sinh, Sleef_finz_sinhf8_u35avx2, Sleef_finz_sinhd4_u35avx2)
DISPATCH_ALL_SKIP_HWY(Cosh, std::cosh, Sleef_finz_coshf8_u10avx2, Sleef_finz_coshd4_u10avx2)
DISPATCH_ALL_LOW_PRECISION(Cosh, Sleef_finz_coshf8_u35avx2, Sleef_finz_coshd4_u35avx2)
DISPATCH_ALL(Tanh, std::tanh, Sleef_finz_tanhf8_u10avx2, Sleef_finz_tanhd4_u10avx2)
DISPATCH_ALL_LOW_PRECISION(Tanh, Sleef_finz_tanhf8_u35avx2, Sleef_finz_tanhd4_u35avx2)

DISPATCH_ALL(Asinh, std::asinh, Sleef_finz_asinhf8_u10avx2, Sleef_finz_asinhd4_u10avx2)
DISPATCH_ALL(Acosh, std::acosh, Sleef_finz_acoshf8_u10avx2, Sleef_finz_acoshd4_u10avx2)
DISPATCH_ALL(Atanh, std::atanh, Sleef_finz_atanhf8_u10avx2, Sleef_finz_atanhd4_u10avx2)

DISPATCH_ALL(Erf, std::erf, Sleef_finz_erff8_u10avx2, Sleef_finz_erfd4_u10avx2)
DISPATCH_ALL(Erfc, std::erfc, Sleef_finz_erfcf8_u15avx2, Sleef_finz_erfcd4_u15avx2)
DISPATCH_ALL(Gamma, std::tgamma, Sleef_finz_tgammaf8_u10avx2, Sleef_finz_tgammad4_u10avx2)
DISPATCH_ALL(LogGamma, std::lgamma, Sleef_finz_lgammaf8_u10avx2, Sleef_finz_lgammad4_u10avx2)
// clang-format on
#endif