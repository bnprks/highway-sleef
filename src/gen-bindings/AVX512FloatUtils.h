#if defined(HIGHWAY_HWY_CONTRIB_SLEEF_AVX512_FLOAT_UTILS_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_SLEEF_AVX512_FLOAT_UTILS_
#undef HIGHWAY_HWY_CONTRIB_SLEEF_AVX512_FLOAT_UTILS_
#else
#define HIGHWAY_HWY_CONTRIB_SLEEF_AVX512_FLOAT_UTILS_
#endif

#include "hwy/highway.h"

#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

HWY_API Vec512<float> GetExponent(Vec512<float> x) {
  return Vec512<float>{_mm512_getexp_ps(x.raw)};
}
HWY_API Vec256<float> GetExponent(Vec256<float> x) {
  return Vec256<float>{_mm256_getexp_ps(x.raw)};
}
template <size_t N>
HWY_API Vec128<float, N> GetExponent(Vec128<float, N> x) {
  return Vec128<float, N>{_mm_getexp_ps(x.raw)};
}

HWY_API Vec512<double> GetExponent(Vec512<double> x) {
  return Vec512<double>{_mm512_getexp_pd(x.raw)};
}
HWY_API Vec256<double> GetExponent(Vec256<double> x) {
  return Vec256<double>{_mm256_getexp_pd(x.raw)};
}
template <size_t N>
HWY_API Vec128<double, N> GetExponent(Vec128<double, N> x) {
  return Vec128<double, N>{_mm_getexp_pd(x.raw)};
}

HWY_API Vec512<float> GetMantissa(Vec512<float> x) {
  return Vec512<float>{
      _mm512_getmant_ps(x.raw, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)};
}
HWY_API Vec256<float> GetMantissa(Vec256<float> x) {
  return Vec256<float>{
      _mm256_getmant_ps(x.raw, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)};
}
template <size_t N>
HWY_API Vec128<float, N> GetMantissa(Vec128<float, N> x) {
  return Vec128<float, N>{
      _mm_getmant_ps(x.raw, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)};
}

HWY_API Vec512<double> GetMantissa(Vec512<double> x) {
  return Vec512<double>{
      _mm512_getmant_pd(x.raw, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)};
}
HWY_API Vec256<double> GetMantissa(Vec256<double> x) {
  return Vec256<double>{
      _mm256_getmant_pd(x.raw, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)};
}
template <size_t N>
HWY_API Vec128<double, N> GetMantissa(Vec128<double, N> x) {
  return Vec128<double, N>{
      _mm_getmant_pd(x.raw, _MM_MANT_NORM_p75_1p5, _MM_MANT_SIGN_nan)};
}

template <int I>
HWY_API Vec512<float> Fixup(Vec512<float> a, Vec512<float> b, Vec512<int> c) {
  return Vec512<float>{_mm512_fixupimm_ps(a.raw, b.raw, c.raw, I)};
}
template <int I>
HWY_API Vec256<float> Fixup(Vec256<float> a, Vec256<float> b, Vec256<int> c) {
  return Vec256<float>{_mm256_fixupimm_ps(a.raw, b.raw, c.raw, I)};
}
template <int I, size_t N>
HWY_API Vec128<float, N> Fixup(Vec128<float, N> a, Vec128<float, N> b,
                               Vec128<int, N> c) {
  return Vec128<float, N>{_mm_fixupimm_ps(a.raw, b.raw, c.raw, I)};
}

template <int I>
HWY_API Vec512<double> Fixup(Vec512<double> a, Vec512<double> b,
                             Vec512<int64_t> c) {
  return Vec512<double>{_mm512_fixupimm_pd(a.raw, b.raw, c.raw, I)};
}
template <int I>
HWY_API Vec256<double> Fixup(Vec256<double> a, Vec256<double> b,
                             Vec256<int64_t> c) {
  return Vec256<double>{_mm256_fixupimm_pd(a.raw, b.raw, c.raw, I)};
}
template <int I, size_t N>
HWY_API Vec128<double, N> Fixup(Vec128<double, N> a, Vec128<double, N> b,
                                Vec128<int64_t, N> c) {
  return Vec128<double, N>{_mm_fixupimm_pd(a.raw, b.raw, c.raw, I)};
}
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif
#endif