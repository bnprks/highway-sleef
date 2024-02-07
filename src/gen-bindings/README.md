## Translation methodology

The overall approach is to use [py-tree-sitter](https://github.com/tree-sitter/py-tree-sitter) to perform code parsing, then use handmade translation data files in `rename_data/` and some custom code to print translations into a generated file (`translate.py`). From a correctness perspective, the main thing to get right is the translations in `rename_data/`, and if those are right the generated code will almost certainly be right.

Sources used by translation code:

-  `sleef/src/libm/sleefsimdsp.c` - All single-precision math operations expressed in SLEEF's internal SIMD abstraction
- `sleef/src/common/df.h` - Some [double-double precision](https://en.wikipedia.org/wiki/Quadruple-precision_floating-point_format#Double-double_arithmetic) math with helper functions when SLEEF needs a bit more than single-precision for intermediate computations
- `sleef/src/common/misc.h` - Constant definitions from Sleef

Files in `rename_data`:

- `simd_ops.tsv` - This is the core translation logic, specifying how C snippets and parameters should be translated into one or more Hwy function calls
  - Note that `df`, `du`, `di` are special variable names denoting HWY vector type tags of floating-point, unsigned int, and signed int respectively. (Size-matched to float/double for current function)
  - `di32` and `du32` are always 32-bit signed/unsiged integers respectively, but are currently not used in an attempt to
    avoid SLEEF's use of mixed-width types for double-precision math
- `function_renames.tsv` - This lists top-level or helper functions from SLEEF that should be translated fully into Hwy. For some small functions, it's a judgement call whether to put them as a function rename or as a SIMD op which will be translated inline
- `types.tsv` - Translations from SLEEF types to Hwy types
- `macro_conditionals.tsv` - Translations for macros used in SLEEF. Outputs of 0 or 1 will cause dead code branches to be eliminated in the translation
- `constant_renames.tsv` - Renames for constants that SLEEF defines in `src/common/misc.h` as macros, so they can be defined as constexpr constants in C++

All the files in `rename_data` are basically TSVs with empty lines ignored and  `#` for end-of-line comments in order to improve human-readability.

Miscellaneous extra bits:

- `translate.py` has a big template string for the output, which also includes a bit of hard-coded stuff that includes Estrin polynomial functions and a hard-coded data table that SLEEF uses for certain trig-function range reductions
  - Note that the Estrin polynomials are modified to take power-of-two powers of x as parameters like SLEEF uses for its `POLY*()` macros