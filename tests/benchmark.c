#define SOKOL_TIME_IMPL
#include "sokol_time.h"
#include <stdio.h>
#include <float.h>
#include <math.h>


#include "../math_intrinsics.h"

//----------------------------------------------------------------------------------------------------------------------
// functions pointer definition
typedef float (*reference_function)(float);
typedef float (*reference_function2)(float, float);
#ifdef __MATH__INTRINSICS__AVX__
    typedef __m256 (*approximation_function)(__m256);
    typedef __m256 (*approximation_function2)(__m256, __m256);
    #define simd_vector_width (8)
#else
    typedef float32x4_t (*approximation_function)(float32x4_t);
    typedef float32x4_t (*approximation_function2)(float32x4_t, float32x4_t);
    #define simd_vector_width (4)
#endif

#define NUM_ITERATIONS (200000000)

//----------------------------------------------------------------------------------------------------------------------
int benchmark(approximation_function function, reference_function reference, const char* name)
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

    for(uint32_t i=0; i<(NUM_ITERATIONS / simd_vector_width); ++i)
    {
        result = _mm256_add_ps(result, function(input));
        input = _mm256_add_ps(input, step);
    }

    output = _mm256_cvtss_f32(result);
#else
    float32x4_t step = vdupq_n_f32(FLT_EPSILON);
    float32x4_t input = vld1q_f32(init_array);
    float32x4_t result = vdupq_n_f32(0.f);

     start = stm_now();

    for(uint32_t i=0; i<(NUM_ITERATIONS / simd_vector_width); ++i)
    {
        result = vaddq_f32(result, function(input));
        input = vaddq_f32(input, step);
    }

    output = vgetq_lane_f32(result, 0);
#endif

    float simd_time = stm_ms(stm_since(start));

    printf(".%s:\t %05.2fms", name, simd_time);

    float total = simd_time / 1000.f;

    start = stm_now();

    for(uint32_t i=0; i<NUM_ITERATIONS; ++i)
        total += reference(total);

    output += total;

    float clib_time = stm_ms(stm_since(start));
    
    printf("\tc std func: %05.2fms\tratio: %2.2fx\n", clib_time, clib_time/simd_time);

    return output;
}

//----------------------------------------------------------------------------------------------------------------------
int benchmark2(approximation_function2 function, reference_function2 reference, const char* name)
{
    float array_x[simd_vector_width];
    float array_y[simd_vector_width];
    uint64_t start = 0;
    int output = 0;

    for(uint32_t i=0; i<simd_vector_width; ++i)
    {
        array_x[i] = (float) (i) / (float) (simd_vector_width);
        array_y[i] = (float) (i); 
    }

#ifdef __MATH__INTRINSICS__AVX__
    __m256 step = _mm256_set1_ps(FLT_EPSILON);
    __m256 v_x = _mm256_loadu_ps(array_x);
    __m256 v_y = _mm256_loadu_ps(array_y);
    __m256 result = _mm256_setzero_ps();

     start = stm_now();

    for(uint32_t i=0; i<(NUM_ITERATIONS / simd_vector_width); ++i)
    {
        result = _mm256_add_ps(result, function(v_x, v_y));
        v_x = _mm256_add_ps(v_x, step);
    }

    output = _mm256_cvtss_f32(result);
#else
    float32x4_t step = vdupq_n_f32(FLT_EPSILON);
    float32x4_t v_x = vld1q_f32(array_x);
    float32x4_t v_y = vld1q_f32(array_y);
    float32x4_t result = vdupq_n_f32(0.f);

     start = stm_now();

    for(uint32_t i=0; i<(NUM_ITERATIONS / simd_vector_width); ++i)
    {
        result = vaddq_f32(result, function(v_x, v_y));
        v_x = vaddq_f32(v_x, step);
    }

    output = vgetq_lane_f32(result, 0);
#endif

    float simd_time = stm_ms(stm_since(start));

    printf(".%s:\t %05.2fms", name, simd_time);

    float total = simd_time / 1000.f;

    start = stm_now();

    float x = 0.f;

    for(uint32_t i=0; i<NUM_ITERATIONS; ++i)
    {
        total += reference(x, x*2.f);
        x += 0.001f;
    }

    output += total;

    float clib_time = stm_ms(stm_since(start));
    
    printf("\tc std func: %05.2fms\tratio: %2.2fx\n", clib_time, clib_time/simd_time);

    return output;
}

//----------------------------------------------------------------------------------------------------------------------
int main(int argc, char * argv[])
{
    stm_setup();

#ifdef __MATH_INTRINSINCS_FAST__
    printf("benchmark, mode fast, %d iterations\n\n", NUM_ITERATIONS);
#else
    printf("benchmark, mode precision, %d iterations\n\n", NUM_ITERATIONS);
#endif

    int output = 0;
    
#ifdef __MATH__INTRINSICS__AVX__
    output += benchmark(mm256_acos_ps, acosf, "mm256_acos_ps");
    output += benchmark(mm256_asin_ps, asinf, "mm256_asin_ps");
    output += benchmark(mm256_atan_ps, atanf, "mm256_atan_ps");
    output += benchmark2(mm256_atan2_ps, atan2f, "mm256_atan2_ps");
    output += benchmark(mm256_cbrt_ps, cbrtf, "mm256_cbrt_ps");
    output += benchmark(mm256_cos_ps, cosf, "mm256_cos_ps");
    output += benchmark(mm256_sin_ps, sinf, "mm256_sin_ps");
    output += benchmark(mm256_exp_ps, expf, "mm256_exp_ps");
    output += benchmark(mm256_exp2_ps, exp2f, "mm256_exp2_ps");
    output += benchmark(mm256_log_ps, logf, "mm256_log_ps");
    output += benchmark(mm256_log2_ps, log2f, "mm256_log2_ps");
    output += benchmark2(mm256_pow_ps, powf, "mm256_pow_ps");
#else
    output += benchmark(vacosq_f32, acosf, "vacosq_f32");
    output += benchmark(vasinq_f32, asinf, "vasinq_f32");
    output += benchmark(vatanq_f32, atanf, "vatanq_f32");
    output += benchmark2(vatan2q_f32, atan2f, "vatan2q_f32");
    output += benchmark(vcbrtq_f32, cbrtf, "vcbrtq_f32");
    output += benchmark(vcosq_f32, cosf, "vcosq_f32");
    output += benchmark(vsinq_f32, sinf, "vsinq_f32");
    output += benchmark(vexpq_f32, expf, "vexpq_f32");
    output += benchmark(vexp2q_f32, exp2f, "vexp2q_f32");
    output += benchmark(vlogq_f32, logf, "vlogq_f32");
    output += benchmark(vlog2q_f32, log2f, "vlog2q_f32");
    output += benchmark2(vpowq_f32, powf, "vpowq_f32");
#endif

    printf("\n%d\n", output);

    return 0;
}