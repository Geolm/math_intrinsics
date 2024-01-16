# math_intrinsics
One header file library that implement of missing transcendental math functions (cos, sin, acos, and more....) using 100% AVX/Neon instructions (no branching)

# why?
AVX and Neon intrinsics don't provide transcendental math functions. Of course there are already some libraries with those functions but there are usually not free, restricted to one tpe of hardware or with low precision. This library is super easy to integrate, with a precision close to the C math library (see below) and with MIT license.

# how to

It's one-header lib, just define the macro once and include the header

```C
#define __MATH__INTRINSICS__IMPLEMENTATION
#include "math_intrinsics.h"
```

# list of functions supported

# references

