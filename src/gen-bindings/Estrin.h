#if defined(HIGHWAY_HWY_CONTRIB_SLEEF_ESTRIN_) == \
    defined(HWY_TARGET_TOGGLE)  // NOLINT
#ifdef HIGHWAY_HWY_CONTRIB_SLEEF_ESTRIN_
#undef HIGHWAY_HWY_CONTRIB_SLEEF_ESTRIN_
#else
#define HIGHWAY_HWY_CONTRIB_SLEEF_ESTRIN_
#endif

#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace hwy {
namespace HWY_NAMESPACE {
namespace sleef {
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
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T c0, T c1, T c2, T c3,
                                     T c4) {
  return MulAdd(x4, c4, MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T c0, T c1, T c2, T c3,
                                     T c4, T c5) {
  return MulAdd(x4, MulAdd(c5, x, c4),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T c0, T c1, T c2, T c3,
                                     T c4, T c5, T c6) {
  return MulAdd(x4, MulAdd(x2, c6, MulAdd(c5, x, c4)),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T c0, T c1, T c2, T c3,
                                     T c4, T c5, T c6, T c7) {
  return MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2,
                                     T c3, T c4, T c5, T c6, T c7, T c8) {
  return MulAdd(x8, c8,
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2,
                                     T c3, T c4, T c5, T c6, T c7, T c8, T c9) {
  return MulAdd(x8, MulAdd(c9, x, c8),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2,
                                     T c3, T c4, T c5, T c6, T c7, T c8, T c9,
                                     T c10) {
  return MulAdd(x8, MulAdd(x2, c10, MulAdd(c9, x, c8)),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2,
                                     T c3, T c4, T c5, T c6, T c7, T c8, T c9,
                                     T c10, T c11) {
  return MulAdd(x8, MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8)),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2,
                                     T c3, T c4, T c5, T c6, T c7, T c8, T c9,
                                     T c10, T c11, T c12) {
  return MulAdd(
      x8, MulAdd(x4, c12, MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
      MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
             MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2,
                                     T c3, T c4, T c5, T c6, T c7, T c8, T c9,
                                     T c10, T c11, T c12, T c13) {
  return MulAdd(x8,
                MulAdd(x4, MulAdd(c13, x, c12),
                       MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2,
                                     T c3, T c4, T c5, T c6, T c7, T c8, T c9,
                                     T c10, T c11, T c12, T c13, T c14) {
  return MulAdd(x8,
                MulAdd(x4, MulAdd(x2, c14, MulAdd(c13, x, c12)),
                       MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T c0, T c1, T c2,
                                     T c3, T c4, T c5, T c6, T c7, T c8, T c9,
                                     T c10, T c11, T c12, T c13, T c14, T c15) {
  return MulAdd(x8,
                MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                       MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
                MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                       MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T x16, T c0, T c1,
                                     T c2, T c3, T c4, T c5, T c6, T c7, T c8,
                                     T c9, T c10, T c11, T c12, T c13, T c14,
                                     T c15, T c16) {
  return MulAdd(
      x16, c16,
      MulAdd(x8,
             MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                    MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
             MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                    MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T x16, T c0, T c1,
                                     T c2, T c3, T c4, T c5, T c6, T c7, T c8,
                                     T c9, T c10, T c11, T c12, T c13, T c14,
                                     T c15, T c16, T c17) {
  return MulAdd(
      x16, MulAdd(c17, x, c16),
      MulAdd(x8,
             MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                    MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
             MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                    MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)))));
}
template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T x16, T c0, T c1,
                                     T c2, T c3, T c4, T c5, T c6, T c7, T c8,
                                     T c9, T c10, T c11, T c12, T c13, T c14,
                                     T c15, T c16, T c17, T c18) {
  return MulAdd(
      x16, MulAdd(x2, c18, MulAdd(c17, x, c16)),
      MulAdd(x8,
             MulAdd(x4, MulAdd(x2, MulAdd(c15, x, c14), MulAdd(c13, x, c12)),
                    MulAdd(x2, MulAdd(c11, x, c10), MulAdd(c9, x, c8))),
             MulAdd(x4, MulAdd(x2, MulAdd(c7, x, c6), MulAdd(c5, x, c4)),
                    MulAdd(x2, MulAdd(c3, x, c2), MulAdd(c1, x, c0)))));
}

template <class T>
HWY_INLINE HWY_MAYBE_UNUSED T Estrin(T x, T x2, T x4, T x8, T x16, T c0, T c1,
                                     T c2, T c3, T c4, T c5, T c6, T c7, T c8,
                                     T c9, T c10, T c11, T c12, T c13, T c14,
                                     T c15, T c16, T c17, T c18, T c19, T c20) {
  return MulAdd(
      x16,
      MulAdd(x4, c20, MulAdd(x2, MulAdd(x, c19, c18), MulAdd(x, c17, c16))),
      MulAdd(x8,
             MulAdd(x4, MulAdd(x2, MulAdd(x, c15, c14), MulAdd(x, c13, c12)),
                    MulAdd(x2, MulAdd(x, c11, c10), MulAdd(x, c9, c8))),
             MulAdd(x4, MulAdd(x2, MulAdd(x, c7, c6), MulAdd(x, c5, c4)),
                    MulAdd(x2, MulAdd(x, c3, c2), MulAdd(x, c1, c0)))));
}

}  // namespace sleef
}  // namespace HWY_NAMESPACE
}  // namespace hwy
HWY_AFTER_NAMESPACE();
#endif