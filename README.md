# math_intrinsics
One header file library that implement missing transcendental math functions (cos, sin, acos, and more....) using 100% AVX/Neon instructions (no branching)

### unit tests build status
[![Build Status](https://github.com/geolm/math_intrinsics/actions/workflows/cmake-multi-platform.yml/badge.svg)](https://github.com/geolm/math_intrinsics/actions)

# why
AVX and Neon intrinsics don't provide transcendental math functions. Of course there are already some libraries with those functions but there are usually not free, restricted to one specific  hardware or with low precision. This library is super easy to integrate, with a precision close to the C math library (see below) and with MIT license.

# how to

It's a one-header lib, just define the macro once in your project and include the header.

```C
#define __MATH__INTRINSICS__IMPLEMENTATION__
#include "math_intrinsics.h"
```

On intel/AMD computer, you need to compile with **-mavx2**. You can add also -mfma. 
On ARM based computer nothing required as the lib is for AArch64


You can define this macro to generate faster albeit less precise functions (see below for more details) :
```C
#define __MATH_INTRINSINCS_FAST__
```

# functions

```C
// max error : 5.960464478e-08
__m256 mm256_cos_ps(__m256 a);

// max error : 5.960464478e-08
__m256 mm256_sin_ps(__m256 a);

// max error : 5.960464478e-08
void mm256_sincos_ps(__m256 a, __m256 *s, __m256 *c);

// max error : 2.384185791e-07
__m256 mm256_acos_ps(__m256 a);

// max error : 1.192092896e-07
__m256 mm256_asin_ps(__m256 a);

// max error : 1.192092896e-07
__m256 mm256_atan_ps(__m256 a);

// max error : 2.384185791e-07
__m256 mm256_atan2_ps(__m256 x, __m256 y);

// max error : 9.107976950e-08
__m256 mm256_log_ps(__m256 a);

// max error : 2.349663504e-07
__m256 mm256_log2_ps(__m256 x);

// max error : 1.108270880e-07
__m256 mm256_exp_ps(__m256 a);

// max error : 1.042427087e-07
__m256 mm256_exp2_ps(__m256 x);

// max error : 1.184910232e-07
__m256 mm256_cbrt_ps(__m256 a);

// max error : 9.768706377e-07
__m256 mm256_pow_ps(__m256 x, __m256 y);
```

Note : the same functions are defined in NEON intrinsics style :

```C
// max error : 5.960464478e-08
float32x4_t vcosq_f32(float32x4_t a);

// max error : 5.960464478e-08
float32x4_t vsinq_f32(float32x4_t a);

// max error : 5.960464478e-08
void vsincosq_f32(float32x4_t a, float32x4_t *s, float32x4_t *c);

// max error : 2.384185791e-07
float32x4_t vacosq_f32(float32x4_t a);

// max error : 1.192092896e-07
float32x4_t vasinq_f32(float32x4_t a);

// max error : 1.192092896e-07
float32x4_t vatanq_f32(float32x4_t a);

// max error : 2.384185791e-07
float32x4_t vatan2q_f32(float32x4_t x, float32x4_t y);

// max error : 9.107976950e-08
float32x4_t vlogq_f32(float32x4_t a);

// max error : 2.349663504e-07
float32x4_t vlog2q_f32(float32x4_t x);

// max error : 1.108270880e-07
float32x4_t vexpq_f32(float32x4_t a);

// max error : 1.042427087e-07
float32x4_t vexp2q_f32(float32x4_t a);

// max error : 1.184910232e-07
float32x4_t vcbrtq_f32(float32x4_t a);

// max error : 9.768706377e-07
float32x4_t vpowq_f32(float32x4_t x, float32x4_t y);

```

# fast functions 

If you use the macro \_\_MATH_INTRINSINCS_FAST\_\_ some functions will have a bit less precision but better performances:

* sin, max_error : 2.682209015e-07 perf : ~1.5x
* cos, max_error : 5.811452866e-07 perf : ~1.5x
* acos, max_error : 6.520748138e-05 perf : ~1.6x
* asin, max_error : 6.520736497e-05 perf : ~1.4x
* exp2, max_error : 2.674510370e-06 perf : ~1.9x
* pow, max error : 8.886078831e-06 perf : ~1.9x

Check the benchmark actions in build system for more details. As you can see, we maintained good precision with a noticeable performance boost. Most  programs could use the fast version.

# FAQ

## is it fast?
The goal of this library is to provide math function with a good precision with every computation done in AVX/NEON. Performance is not the focus.

Here's the benchmark results on my old Intel Core i7 from 2018 for 1 billion of operations, comparison against the C standard library.

```C
benchmark : mode precision

.mm256_acos_ps: 723.730 ms	 c std func: 5408.153 ms	  ratio: 7.47x
.mm256_asin_ps: 692.439 ms	 c std func: 5419.091 ms	  ratio: 7.83x
.mm256_atan_ps: 733.843 ms	 c std func: 3762.987 ms	  ratio: 5.13x
.mm256_cbrt_ps: 1522.731 ms	 c std func: 19559.201 ms	  ratio: 12.84x
.mm256_cos_ps: 882.112 ms        c std func: 15540.117 ms	  ratio: 17.62x
.mm256_sin_ps: 838.590 ms	 c std func: 15214.896 ms	  ratio: 18.14x
.mm256_exp_ps: 830.130 ms	 c std func: 4399.218 ms	  ratio: 5.30x
.mm256_exp2_ps: 1007.015 ms	 c std func: 2076.871 ms	  ratio: 2.06x
.mm256_log_ps: 1019.277 ms	 c std func: 16832.281 ms	  ratio: 16.51x
.mm256_log2_ps: 479.116 ms	 c std func: 3594.876 ms	  ratio: 7.50x
```

Don't forget : the function mm256_sincos_ps computes sinus and cosinus for the cost of one. Also you can use the macro \_\_MATH_INTRINSINCS_FAST\_\_ 

## why AVX2 ?

On multiple functions this library use a float as an int to have access to the mantissa and the exponent part. While it's doable with AVX1 using SSE4.2, I don't see the point of not using AVX2 which have been on CPU since 2013.

## does it handle all float cases (+inf, -inf, NAN) as the C math lib?

Yes, all functions (except atan2 and pow) are compliant to +inf, -inf, NAN and other special cases (for example log(-4) == NAN). All based on the doc found here https://en.cppreference.com/w/

## what's tested?

The unit tests cover precision and special cases (inf, nan, ...). At the moment, the Neon version is not ran on GitHub but rather manually on my M1 Pro machine as I didn't had time to setup the emulator properly. 

# references

[cephes math library](https://github.com/jeremybarnes/cephes/blob/master/single/)

[simple SSE sin/cos](http://gruntthepeon.free.fr/ssemath/)

[speeding up atan2f by 50x](https://mazzo.li/posts/vectorized-atan2.html)
