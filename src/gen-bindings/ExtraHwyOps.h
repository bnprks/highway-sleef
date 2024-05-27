#if defined(HIGHWAY_HWY_CONTRIB_SLEEF_AVX512_FLOAT_UTILS_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_SLEEF_AVX512_FLOAT_UTILS_
#undef HIGHWAY_HWY_CONTRIB_SLEEF_AVX512_FLOAT_UTILS_
#else
#define HIGHWAY_HWY_CONTRIB_SLEEF_AVX512_FLOAT_UTILS_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {

// GetExponent and GetMantissa for AVX512 architectures
#if HWY_ARCH_X86 && HWY_TARGET <= HWY_AVX3
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
#endif

// Faster NearestInt f32->i32 and f64->i64
// Results that cannot fit into an i32 are undefined, *even for the i64 versions*
// The SLEEF algorithms are already designed around these constraints.
#if HWY_ARCH_X86
#if HWY_TARGET <= HWY_AVX3
HWY_API Vec512<int32_t> NearestIntFast(const Vec512<float> v) {
  return Vec512<int32_t>{_mm512_cvtps_epi32(v.raw)};
}
HWY_API Vec512<int64_t> NearestIntFast(const Vec512<double> v) {
  return Vec512<int64_t>{_mm512_cvtpd_epi64(v.raw)};
}
HWY_API Vec256<int64_t> NearestIntFast(const Vec256<double> v) {
  return Vec256<int64_t>{_mm256_cvtpd_epi64(v.raw)};
}
template <size_t N>
HWY_API Vec128<int64_t, N> NearestIntFast(const Vec128<double, N> v) {
  const RebindToSigned<DFromV<decltype(v)>> di;
  return VFromD<decltype(di)>{_mm_cvtps_epi32(v.raw)};
}
#endif // HWY_TARGET <= HWY_AVX3

#if HWY_TARGET <= HWY_AVX2
HWY_API Vec256<int32_t> NearestIntFast(const Vec256<float> v) {
  return Vec256<int32_t>{_mm256_cvtps_epi32(v.raw)};
}
#if HWY_TARGET > HWY_AVX3
HWY_API Vec256<int64_t> NearestIntFast(const Vec256<double> v) {
  return Vec256<int64_t>{_mm256_cvtepi32_epi64(_mm256_cvtpd_epi32(v.raw))};
}
#endif // HWY_TARGET > HWY_AVX3
#endif // HWY_TARGET <= HWY_AVX2

#if HWY_TARGET <= HWY_SSE4 && HWY_TARGET > HWY_AVX3
template <size_t N>
HWY_API Vec128<int64_t, N> NearestIntFast(const Vec128<double, N> v) {
  return Vec128<int64_t, N>{_mm_cvtepi32_epi64(_mm_cvtpd_epi32(v.raw))};
}
#elif HWY_TARGET > HWY_AVX3
template <size_t N>
HWY_API Vec128<int64_t, N> NearestIntFast(const Vec128<double, N> v) {
  const RebindToSigned<DFromV<decltype(v)>> di;
  return ConvertTo(di, Round(v));
}
#endif

#if HWY_TARGET <= HWY_SSE2
template <size_t N>
HWY_API Vec128<int32_t, N> NearestIntFast(const Vec128<float, N> v) {
  const RebindToSigned<DFromV<decltype(v)>> di;
  return VFromD<decltype(di)>{_mm_cvtps_epi32(v.raw)};
}
#else
template <size_t N>
HWY_API Vec128<int32_t, N> NearestIntFast(const Vec128<float, N> v) {
  const RebindToSigned<DFromV<decltype(v)>> di;
  return ConvertTo(di, Round(v));
}
#endif

#else // HWY_ARCH_X86
template <class VF, class DI = RebindToSigned<DFromV<VF>>, HWY_IF_I32_D(DI)>
HWY_API VFromD<DI> NearestIntFast(VF v) {
  return NearestInt(v);
}
template <class VF, class DI = RebindToSigned<DFromV<VF>>, HWY_IF_I64_D(DI)>
HWY_API VFromD<DI> NearestIntFast(VF v) {
  return ConvertTo(DI(), Round(v));
}
#endif // HWY_ARCH_X86

}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();

#endif