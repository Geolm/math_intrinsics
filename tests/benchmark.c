#define SOKOL_TIME_IMPL
#include "sokol_time.h"
#include <stdio.h>
#include <float.h>
#include <math.h>


#include "../math_intrinsics.h"

typedef float (*reference_function)(float);
#ifdef __MATH__INTRINSICS__AVX__
    typedef __m256 (*approximation_function)(__m256);
    #define simd_vector_width (8)
#else
    typedef float32x4_t (*approximation_function)(float32x4_t);
    #define simd_vector_width (4)
#endif

#define NUM_ITERATIONS (1000000000)

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

    printf(".%s: %3.3f ms", name, simd_time);

    float total = simd_time;

    start = stm_now();

    for(uint32_t i=0; i<NUM_ITERATIONS; ++i)
        total += reference(total);

    output += total;

    float clib_time = stm_ms(stm_since(start));
    
    printf("\t c std func: %3.3f ms\tratio: %2.2fx\n", clib_time, clib_time/simd_time);

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
    output += benchmark(mm256_acos_ps, acosf, "mm256_acos_ps");
    output += benchmark(mm256_asin_ps, asinf, "mm256_asin_ps");
    output += benchmark(mm256_atan_ps, atanf, "mm256_atan_ps");
    output += benchmark(mm256_cbrt_ps, cbrtf, "mm256_cbrt_ps");
    output += benchmark(mm256_cos_ps, cosf, "mm256_cos_ps");
    output += benchmark(mm256_sin_ps, sinf, "mm256_sin_ps");
    output += benchmark(mm256_exp_ps, expf, "mm256_exp_ps");
    output += benchmark(mm256_exp2_ps, exp2f, "mm256_exp2_ps");
    output += benchmark(mm256_log_ps, logf, "mm256_log_ps");
    output += benchmark(mm256_log2_ps, log2f, "mm256_log2_ps");
#else
    output += benchmark(vacosq_f32, acosf, "vacosq_f32");
    output += benchmark(vasinq_f32, asinf, "vasinq_f32");
    output += benchmark(vatanq_f32, atanf, "vatanq_f32");
    output += benchmark(vcbrtq_f32, cbrtf, "vcbrtq_f32");
    output += benchmark(vcosq_f32, cosf, "vcosq_f32");
    output += benchmark(vsinq_f32, sinf, "vsinq_f32");
    output += benchmark(vexpq_f32, expf, "vexpq_f32");
    output += benchmark(vexp2q_f32, exp2f, "vexp2q_f32");
    output += benchmark(vlogq_f32, logf, "vlogq_f32");
    output += benchmark(vlog2q_f32, log2f, "vlog2q_f32");
#endif

    printf("\n%d\n", output);

    return 0;
}