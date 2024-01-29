#define SOKOL_TIME_IMPL
#include "sokol_time.h"
#include <stdio.h>
#include <float.h>


#include "../math_intrinsics.h"

#ifdef __MATH__INTRINSICS__AVX__
    typedef __m256 (*approximation_function)(__m256);
    #define simd_vector_width (8)
#else
    typedef float32x4_t (*approximation_function)(float32x4_t);
    #define simd_vector_width (4)
#endif

#define NUM_ITERATIONS (1000000000 / simd_vector_width)

int benchmark(approximation_function function, const char* name)
{
    float init_array[simd_vector_width];
    uint64_t start = 0;
    int output = 0;

    for(uint32_t i=0; i<simd_vector_width; ++i)
        init_array[i] = (float) (i) / (float) (simd_vector_width);

#ifdef __MATH__INTRINSICS__AVX__
    __m256 step = _mm256_set1_ps(FLT_EPSILON);
    __m256 input = _mm256_loadu_ps(init_array);
    __m256 result = _mm256_setzero_ps();

     start = stm_now();

    for(uint32_t i=0; i<NUM_ITERATIONS; ++i)
    {
        result = _mm256_add_ps(result, function(input));
        input = _mm256_add_ps(input, step);
    }

    output = _mm256_cvtss_f32(result);
#else

#endif

    printf(".%s: %3.3f ms \n", name, stm_ms(stm_since(start)));

    return output;
}


int main(int argc, char * argv[])
{
    stm_setup();

#ifdef __MATH_INTRINSINCS_FAST__
    printf("benchmark : mode fast\n\n");
#else
    printf("benchmark : mode precision\n\n");
#endif

    int output = 0;
    
#ifdef __MATH__INTRINSICS__AVX__
    output += benchmark(mm256_acos_ps, "mm256_acos_ps");
    output += benchmark(mm256_asin_ps, "mm256_asin_ps");
    output += benchmark(mm256_atan_ps, "mm256_atan_ps");
    output += benchmark(mm256_cbrt_ps, "mm256_cbrt_ps");
    output += benchmark(mm256_cos_ps, "mm256_cos_ps");
    output += benchmark(mm256_sin_ps, "mm256_sin_ps");
    output += benchmark(mm256_exp_ps, "mm256_exp_ps");
    output += benchmark(mm256_exp2_ps, "mm256_exp2_ps");
    output += benchmark(mm256_log_ps, "mm256_log_ps");
    output += benchmark(mm256_log2_ps, "mm256_log2_ps");
#else

#endif

    return (output!=0);
}