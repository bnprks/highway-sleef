#ifndef DEFINE_DISPATCHERS_H_
#define DEFINE_DISPATCHERS_H_

#include "sleef.h"
#include "hwy/contrib/math/math-inl.h"
#include "src/gen-bindings/sleef-generated.h"

// Define functions with signature f(const float *in, size_t n, float *__restrict__ out)
// These all are defined just for AVX2
// Each op has a *Sleef, *Hwy, and *Translated variant

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

#endif