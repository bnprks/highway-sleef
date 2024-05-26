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


// Register an AVX2 highway wrapper for a Sleef function
#define SLEEF_TO_HWY_D4(NAME, FN)                \
  HWY_BEFORE_NAMESPACE();                        \
  namespace hwy {                                \
  namespace N_AVX2 {                             \
  template <class D>                             \
  HWY_INLINE Vec<D> NAME(const D df, Vec<D> d) { \
    return Vec256<double>{FN(d.raw)};             \
  }                                              \
  }                                              \
  }                                              \
  HWY_AFTER_NAMESPACE();

#define WRAP_OP1F(NAME, FN)                                        \
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

#define WRAP_OP1D(NAME, FN)                                        \
  HWY_BEFORE_NAMESPACE();                                         \
  namespace hwy {                                                 \
  namespace HWY_NAMESPACE {                                       \
  void NAME(const double *in, size_t n, double *__restrict__ out) { \
    using D = ScalableTag<double>;                                 \
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

#define DISPATCH_OP1F(NAME, FN)                                                 \
  namespace hwy {                                                              \
  HWY_NOINLINE void NAME(const float *in, size_t n, float *__restrict__ out) { \
    HWY_STATIC_DISPATCH(NAME)(in, n, out);                                     \
  }                                                                            \
  }

#define DISPATCH_OP1D(NAME, FN)                                                 \
  namespace hwy {                                                              \
  HWY_NOINLINE void NAME(const double *in, size_t n, double *__restrict__ out) { \
    HWY_STATIC_DISPATCH(NAME)(in, n, out);                                     \
  }                                                                            \
  }

#define DISPATCH_AND_WRAP_OP1F(NAME, FN) \
  WRAP_OP1F(NAME, FN)                    \
  DISPATCH_OP1F(NAME, FN)

#define DISPATCH_AND_WRAP_OP1D(NAME, FN) \
  WRAP_OP1D(NAME, FN)                    \
  DISPATCH_OP1D(NAME, FN)

// Define dispatch names based on common naming expectations for operations
#define DISPATCH_ALL_F(OP, SLEEF_NAME)               \
  DISPATCH_AND_WRAP_OP1F(OP##f##Hwy, OP)               \
  DISPATCH_AND_WRAP_OP1F(OP##f##Translated, sleef::OP) \
  SLEEF_TO_HWY_F8(OP##f##Sleef, SLEEF_NAME)           \
  DISPATCH_AND_WRAP_OP1F(OP##f##Sleef, OP##f##Sleef)

#define DISPATCH_ALL_D(OP, SLEEF_NAME)               \
  DISPATCH_AND_WRAP_OP1D(OP##d##Hwy, OP)               \
  DISPATCH_AND_WRAP_OP1D(OP##d##Translated, sleef::OP) \
  SLEEF_TO_HWY_D4(OP##d##Sleef, SLEEF_NAME)           \
  DISPATCH_AND_WRAP_OP1D(OP##d##Sleef, OP##d##Sleef)

#define DISPATCH_ALL_SKIP_HWY_F(OP, SLEEF_NAME) \
  DISPATCH_AND_WRAP_OP1F(OP##f##Translated, sleef::OP) \
  SLEEF_TO_HWY_F8(OP##f##Sleef, SLEEF_NAME)           \
  DISPATCH_AND_WRAP_OP1F(OP##f##Sleef, OP##f##Sleef)

#define DISPATCH_ALL_SKIP_HWY_D(OP, SLEEF_NAME) \
  DISPATCH_AND_WRAP_OP1D(OP##d##Translated, sleef::OP) \
  SLEEF_TO_HWY_D4(OP##d##Sleef, SLEEF_NAME)           \
  DISPATCH_AND_WRAP_OP1D(OP##d##Sleef, OP##d##Sleef)

// Define dispatch names for low-precision variants which also have a high
// precision variant
#define DISPATCH_ALL_LOW_PRECISION_F(OP, SLEEF_NAME)           \
  DISPATCH_AND_WRAP_OP1F(OP##Fast##f##Translated, sleef::OP##Fast) \
  SLEEF_TO_HWY_F8(OP##Fast##f##Sleef, SLEEF_NAME)                 \
  DISPATCH_AND_WRAP_OP1F(OP##Fast##f##Sleef, OP##Fast##f##Sleef)

#define DISPATCH_ALL_LOW_PRECISION_D(OP, SLEEF_NAME)           \
  DISPATCH_AND_WRAP_OP1D(OP##Fast##d##Translated, sleef::OP##Fast) \
  SLEEF_TO_HWY_D4(OP##Fast##d##Sleef, SLEEF_NAME)                 \
  DISPATCH_AND_WRAP_OP1D(OP##Fast##d##Sleef, OP##Fast##d##Sleef)

DISPATCH_ALL_F(Exp, Sleef_finz_expf8_u10avx2)
DISPATCH_ALL_F(Expm1, Sleef_finz_expm1f8_u10avx2)
DISPATCH_ALL_F(Log, Sleef_finz_logf8_u10avx2)
DISPATCH_ALL_F(Log1p, Sleef_finz_log1pf8_u10avx2)
DISPATCH_ALL_F(Log2, Sleef_finz_log2f8_u10avx2)

DISPATCH_ALL_F(Sin, Sleef_finz_sinf8_u10avx2)
DISPATCH_ALL_F(Cos, Sleef_finz_cosf8_u10avx2)
DISPATCH_ALL_SKIP_HWY_F(Tan, Sleef_finz_tanf8_u10avx2)
DISPATCH_ALL_LOW_PRECISION_F(Sin, Sleef_finz_sinf8_u35avx2)
DISPATCH_ALL_LOW_PRECISION_F(Cos, Sleef_finz_cosf8_u35avx2)
DISPATCH_ALL_LOW_PRECISION_F(Tan, Sleef_finz_tanf8_u35avx2)

DISPATCH_ALL_F(Sinh, Sleef_finz_sinhf8_u10avx2)
DISPATCH_ALL_SKIP_HWY_F(Cosh, Sleef_finz_coshf8_u10avx2)
DISPATCH_ALL_F(Tanh, Sleef_finz_tanhf8_u10avx2)
DISPATCH_ALL_LOW_PRECISION_F(Sinh, Sleef_finz_sinhf8_u35avx2)
DISPATCH_ALL_LOW_PRECISION_F(Cosh, Sleef_finz_coshf8_u35avx2)
DISPATCH_ALL_LOW_PRECISION_F(Tanh, Sleef_finz_tanhf8_u35avx2)


DISPATCH_ALL_F(Asin, Sleef_finz_asinf8_u10avx2)
DISPATCH_ALL_F(Acos, Sleef_finz_acosf8_u10avx2)
DISPATCH_ALL_F(Atan, Sleef_finz_atanf8_u10avx2)
DISPATCH_ALL_LOW_PRECISION_F(Asin, Sleef_finz_asinf8_u35avx2)
DISPATCH_ALL_LOW_PRECISION_F(Acos, Sleef_finz_acosf8_u35avx2)
DISPATCH_ALL_LOW_PRECISION_F(Atan, Sleef_finz_atanf8_u35avx2)

DISPATCH_ALL_F(Asinh, Sleef_finz_asinhf8_u10avx2)
DISPATCH_ALL_F(Acosh, Sleef_finz_acoshf8_u10avx2)
DISPATCH_ALL_F(Atanh, Sleef_finz_atanhf8_u10avx2)


// Double precision
DISPATCH_ALL_D(Exp, Sleef_finz_expd4_u10avx2)
DISPATCH_ALL_D(Expm1, Sleef_finz_expm1d4_u10avx2)
DISPATCH_ALL_D(Log, Sleef_finz_logd4_u10avx2)
DISPATCH_ALL_D(Log1p, Sleef_finz_log1pd4_u10avx2)
DISPATCH_ALL_D(Log2, Sleef_finz_log2d4_u10avx2)



DISPATCH_ALL_D(Sin, Sleef_finz_sind4_u10avx2)
DISPATCH_ALL_D(Cos, Sleef_finz_cosd4_u10avx2)
DISPATCH_ALL_SKIP_HWY_D(Tan, Sleef_finz_tand4_u10avx2)
DISPATCH_ALL_LOW_PRECISION_D(Sin, Sleef_finz_sind4_u35avx2)
DISPATCH_ALL_LOW_PRECISION_D(Cos, Sleef_finz_cosd4_u35avx2)
DISPATCH_ALL_LOW_PRECISION_D(Tan, Sleef_finz_tand4_u35avx2)


DISPATCH_ALL_D(Sinh, Sleef_finz_sinhd4_u10avx2)
DISPATCH_ALL_SKIP_HWY_D(Cosh, Sleef_finz_coshd4_u10avx2)
DISPATCH_ALL_D(Tanh, Sleef_finz_tanhd4_u10avx2)
DISPATCH_ALL_LOW_PRECISION_D(Sinh, Sleef_finz_sinhd4_u35avx2)
DISPATCH_ALL_LOW_PRECISION_D(Cosh, Sleef_finz_coshd4_u35avx2)
DISPATCH_ALL_LOW_PRECISION_D(Tanh, Sleef_finz_tanhd4_u35avx2)


DISPATCH_ALL_D(Asin, Sleef_finz_asind4_u10avx2)
DISPATCH_ALL_D(Acos, Sleef_finz_acosd4_u10avx2)
DISPATCH_ALL_D(Atan, Sleef_finz_atand4_u10avx2)
DISPATCH_ALL_LOW_PRECISION_D(Asin, Sleef_finz_asind4_u35avx2)
DISPATCH_ALL_LOW_PRECISION_D(Acos, Sleef_finz_acosd4_u35avx2)
DISPATCH_ALL_LOW_PRECISION_D(Atan, Sleef_finz_atand4_u35avx2)

DISPATCH_ALL_D(Asinh, Sleef_finz_asinhd4_u10avx2)
DISPATCH_ALL_D(Acosh, Sleef_finz_acoshd4_u10avx2)
DISPATCH_ALL_D(Atanh, Sleef_finz_atanhd4_u10avx2)


#endif