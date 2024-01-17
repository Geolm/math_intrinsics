# math_intrinsics
One header file library that implement of missing transcendental math functions (cos, sin, acos, and more....) using 100% AVX/Neon instructions (no branching)

# why?
AVX and Neon intrinsics don't provide transcendental math functions. Of course there are already some libraries with those functions but there are usually not free, restricted to one tpe of hardware or with low precision. This library is super easy to integrate, with a precision close to the C math library (see below) and with MIT license.

# how to

It's one-header lib, just define the macro once and include the header

```C
#define __MATH__INTRINSICS__IMPLEMENTATION__
#include "math_intrinsics.h"
```

# functions

```C
// max error : 5.960464478e-08
__m256 _mm256_cos_ps(__m256 a);

// max error : 5.960464478e-08
__m256 _mm256_sin_ps(__m256 a);

// max error : 5.960464478e-08
void _mm256_sincos_ps(__m256 a, __m256 *s, __m256 *c);

// max error : 2.384185791e-07
__m256 _mm256_acos_ps(__m256 a);

// max error : 1.192092896e-07
__m256 _mm256_asin_ps(__m256 a);

// max error : 6.699562073e-05
__m256 _mm256_atan_ps(__m256 a);

// max error : 2.384185791e-07
__m256 _mm256_atan2_ps(__m256 x, __m256 y);

// max error : 4.768371582e-07
__m256 _mm256_log_ps(__m256 a);

// max error : 1.108270880e-07
__m256 _mm256_exp_ps(__m256 a);

// max error : 4.768371582e-07
__m256 _mm256_cbrt_ps(__m256 a);
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

// max error : 6.699562073e-05
float32x4_t vatanq_f32(float32x4_t a);

// max error : 2.384185791e-07
float32x4_t vatan2q_f32(float32x4_t x, float32x4_t y);

// max error : 4.768371582e-07
float32x4_t vlogq_f32(float32x4_t a);

// max error : 1.108270880e-07
float32x4_t vexpq_f32(float32x4_t a);

// max error : 4.768371582e-07
float32x4_t vcbrtq_f32(float32x4_t a);
```

# references

[cephes math library](https://github.com/jeremybarnes/cephes/blob/master/single/)

[simple SSE sin/cos](http://gruntthepeon.free.fr/ssemath/)

[speeding up atan2f by 50x](https://mazzo.li/posts/vectorized-atan2.html)

