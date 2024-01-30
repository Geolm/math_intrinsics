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
```

# fast functions 

If you use the macro \_\_MATH_INTRINSINCS_FAST\_\_ some functions will have less precision but better performances:

* sin, max_error : 2.682209015e-07
* cos, max_error : 5.811452866e-07
* acos, max_error : 6.520748138e-05
* asin, max_error : 6.520736497e-05
* atan, max_error : 7.289648056e-05
* atan2, max_error : 8.535385132e-05
* exp2, max_error : 2.317290893e-07
* cbrt, max_error : 9.659048374e-05


# references

[cephes math library](https://github.com/jeremybarnes/cephes/blob/master/single/)

[simple SSE sin/cos](http://gruntthepeon.free.fr/ssemath/)

[speeding up atan2f by 50x](https://mazzo.li/posts/vectorized-atan2.html)

# FAQ

## is it fast?
The goal of this library is to provide math function with a good precision with every computation done in AVX/NEON. Performance is not the focus.

Here's the benchmark results on my old Intel Core i7 from 2018 for 10 billions of operations

### precision mode

* mm256_acos_ps: 7795.786 ms
* mm256_asin_ps: 7034.068 ms 
* mm256_atan_ps: 7797.666 ms 
* mm256_cbrt_ps: 15130.169 ms 
* mm256_cos_ps: 8600.893 ms 
* mm256_sin_ps: 8288.432 ms 
* mm256_exp_ps: 8647.793 ms 
* mm256_exp2_ps: 10130.995 ms 
* mm256_log_ps: 10423.453 ms 
* mm256_log2_ps: 5232.928 ms 

### fast mode

Using \_\_MATH_INTRINSINCS_FAST\_\_

* mm256_acos_ps: 4823.037 ms 
* mm256_asin_ps: 4982.991 ms 
* mm256_atan_ps: 7213.156 ms 
* mm256_cbrt_ps: 14716.824 ms 
* mm256_cos_ps: 5441.888 ms 
* mm256_sin_ps: 5186.748 ms 
* mm256_exp_ps: 8429.838 ms 
* mm256_exp2_ps: 5262.944 ms 
* mm256_log_ps: 10318.204 ms 
* mm256_log2_ps: 5130.680 ms 


## why AVX2 ?

On multiple functions this library use a float as an int to have access to the mantissa and the exponent part. While it's doable with AVX1 using SSE4.2, I don't see the point of not using AVX2 which have been on intel CPU since 2013.

## does it handle all float cases (+inf, -inf, NAN) as the C math lib?

Yes, all functions (even the fast ones) are compliant to +inf, -inf, NAN and other special cases (for example log(-4) == NAN). All based on the doc found here https://en.cppreference.com/w/

## what's tested?

The unit tests cover precision and special cases (inf, nan, ...). At the moment, the Neon version is not ran on GitHub but rather manually on my M1 Pro machine as I didn't had time to setup the emulator properly. 
