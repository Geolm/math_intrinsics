#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <float.h>
#include <stdbool.h>
#include "greatest.h"

#include "../math_intrinsics.h"

//----------------------------------------------------------------------------------------------------------------------
// functions pointer definition
typedef float (*reference_function)(float);
#ifdef __MATH__INTRINSICS__AVX__
    typedef __m256 (*approximation_function)(__m256);
    #define simd_vector_width (8)
#else
    typedef float32x4_t (*approximation_function)(float32x4_t);
    #define simd_vector_width (4)
#endif

//----------------------------------------------------------------------------------------------------------------------
// generic unit test
TEST generic_test(reference_function ref, approximation_function approx, float range_min, float range_max, float epsilon, uint32_t num_elements, bool relative_error, const char* name)
{
    float* input = (float*) malloc(num_elements * sizeof(float));
    float* result = (float*) malloc(num_elements * sizeof(float));
    float step = ((range_max - range_min) / (float) (num_elements-1));
    uint32_t num_vectors = num_elements / simd_vector_width;

    for(uint32_t i=0; i<num_elements; ++i)
    {
        input[i] = (step * (float)(i)) + range_min;
        result[i] = ref(input[i]);
    }

#ifdef __MATH__INTRINSICS__AVX__
    __m256 v_epsilon = _mm256_set1_ps(epsilon);
    __m256 v_max_error = _mm256_setzero_ps();

    for(uint32_t i=0; i<num_vectors; ++i)
    {
        __m256 v_input = _mm256_loadu_ps(input + i * simd_vector_width);
        __m256 v_result = _mm256_loadu_ps(result+ i * simd_vector_width);
        __m256 v_approx = approx(v_input);
        __m256 v_error = _mm256_and_ps(_mm256_sub_ps(v_approx, v_result), _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));

        if (relative_error)
            v_error = _mm256_div_ps(v_error, v_result);
        
        ASSERT(_mm256_movemask_ps(_mm256_cmp_ps(v_error, v_epsilon, _CMP_LE_OQ)) == 0xff);
        v_max_error = _mm256_max_ps(v_max_error, v_error);
    }

    v_max_error = _mm256_max_ps(v_max_error, _mm256_permute_ps(v_max_error, _MM_SHUFFLE(2, 1, 0, 3)));
    v_max_error = _mm256_max_ps(v_max_error, _mm256_permute_ps(v_max_error, _MM_SHUFFLE(1, 0, 3, 2)));
    v_max_error = _mm256_max_ps(v_max_error, _mm256_permute2f128_ps(v_max_error, v_max_error, 1));

    printf("%s max error : %.*e\n", name, FLT_DECIMAL_DIG, _mm256_cvtss_f32(v_max_error));
#else
    float32x4_t v_epsilon = vdupq_n_f32(epsilon);
    float32x4_t v_max_error = vdupq_n_f32(0.f);

    for(uint32_t i=0; i<num_vectors; ++i)
    {
        float32x4_t v_input = vld1q_f32(input + i * simd_vector_width);
        float32x4_t v_result = vld1q_f32(result+ i * simd_vector_width);
        float32x4_t v_approx = approx(v_input);
        float32x4_t v_error = vabsq_f32(vsubq_f32(v_approx, v_result));

        if (relative_error)
            v_error = vdivq_f32(v_error, v_result);
        
        ASSERT(vminvq_u32(vcleq_f32(v_error, v_epsilon)) == UINT32_MAX);
        v_max_error = vmaxq_f32(v_max_error, v_error);
    }

    printf("%s max error : %.*e\n", name, FLT_DECIMAL_DIG, vmaxvq_f32(v_max_error));
#endif

    free(input);
    free(result);
    
    PASS();
}

//----------------------------------------------------------------------------------------------------------------------
float atan2_angle(float angle) {return atan2f(sinf(angle) * angle, cosf(angle) * angle);}

#ifdef __MATH__INTRINSICS__AVX__
__m256 simd_atan2(__m256 angle) {return mm256_atan2_ps(_mm256_mul_ps(mm256_cos_ps(angle), angle), _mm256_mul_ps(mm256_sin_ps(angle), angle));}
#else
float32x4_t simd_atan2(float32x4_t angle) {return vatan2q_f32(vmulq_f32(vcosq_f32(angle), angle), vmulq_f32(vsinq_f32(angle), angle));}
#endif


#define NUM_SAMPLES (1024)

SUITE(trigonometry)
{
    printf(".");

#ifdef __MATH__INTRINSICS__AVX__
    RUN_TESTp(generic_test, sinf, mm256_sin_ps, -10.f, 10.f, FLT_EPSILON, NUM_SAMPLES, false, "mm256_sin_ps");
    RUN_TESTp(generic_test, cosf, mm256_cos_ps, -10.f, 10.f, FLT_EPSILON, NUM_SAMPLES, false, "mm256_cos_ps");
    RUN_TESTp(generic_test, acosf, mm256_acos_ps, -1.f, 1.f, 1.e-06f, NUM_SAMPLES, false, "mm256_acos_ps");
    RUN_TESTp(generic_test, asinf, mm256_asin_ps, -1.f, 1.f, 1.e-06f, NUM_SAMPLES, false, "mm256_asin_ps");
    RUN_TESTp(generic_test, atanf, mm256_atan_ps, -10.f, 10.f, 1.e-04f, NUM_SAMPLES, false, "mm256_atan_ps");
    RUN_TESTp(generic_test, atan2_angle, simd_atan2, FLT_EPSILON, 3.14f, 3.e-07f, NUM_SAMPLES, false, "mm256_atan2_ps");
#else
    RUN_TESTp(generic_test, sinf, vsinq_f32, -10.f, 10.f, FLT_EPSILON, NUM_SAMPLES, false, "vsinq_f32");
    RUN_TESTp(generic_test, cosf, vcosq_f32, -10.f, 10.f, FLT_EPSILON, NUM_SAMPLES, false, "vcosq_f32");
    RUN_TESTp(generic_test, acosf, vacosq_f32, -1.f, 1.f, 1.e-06f, NUM_SAMPLES, false, "vacosq_f32");
    RUN_TESTp(generic_test, asinf, vasinq_f32, -1.f, 1.f, 1.e-06f, NUM_SAMPLES, false, "vasinq_f32");
    RUN_TESTp(generic_test, atanf, vatanq_f32, -10.f, 10.f, 1.e-04f, NUM_SAMPLES, false, "vatanq_f32");
    RUN_TESTp(generic_test, atan2_angle, simd_atan2, FLT_EPSILON, 3.14f, 3.e-07f, NUM_SAMPLES, false, "vatan2q_f32");
#endif
}

SUITE(exponentiation)
{
    printf(".");
#ifdef __MATH__INTRINSICS__AVX__
    RUN_TESTp(generic_test, logf, mm256_log_ps, FLT_EPSILON, 1.e20f, 1.e-07f, 32768, true, "mm256_log_ps");
    RUN_TESTp(generic_test, log2f, mm256_log2_ps, FLT_EPSILON, 1.e20f, 3.e-07f, 32768, true, "mm256_log2_ps");
    RUN_TESTp(generic_test, expf, mm256_exp_ps, -87.f, 87.f, 2.e-07f, NUM_SAMPLES, true, "mm256_exp_ps");
    RUN_TESTp(generic_test, exp2f, mm256_exp2_ps, -126.f, 126.f, 2.e-07f, NUM_SAMPLES, true, "mm256_exp2");
    RUN_TESTp(generic_test, cbrtf, mm256_cbrt_ps, -1000.f, 1000.f, 2.e-07f, 4096, true, "mm256_cbrt_ps");
#else
    RUN_TESTp(generic_test, logf, vlogq_f32, FLT_EPSILON, 1.e20f, 1.e-07f, 32768, true, "vlogq_f32");
    RUN_TESTp(generic_test, log2f, vlog2q_f32, FLT_EPSILON, 1.e20f, 3.e-07f, 32768, true, "vlog2q_f32");
    RUN_TESTp(generic_test, expf, vexpq_f32, -87.f, 87.f, 2.e-07f, NUM_SAMPLES, true, "vexpq_f32");
    RUN_TESTp(generic_test, exp2f, vexp2q_f32, -126.f, 126.f, 2.e-07f, NUM_SAMPLES, true, "vexpq_f32");
    RUN_TESTp(generic_test, cbrtf, vcbrtq_f32, -1000.f, 1000.f, 2.e-07f, 4096, true, "vcbrtq_f32");
#endif
}

GREATEST_MAIN_DEFS();

int main(int argc, char * argv[])
{
    GREATEST_MAIN_BEGIN();

    RUN_SUITE(trigonometry);
    RUN_SUITE(exponentiation);

    GREATEST_MAIN_END();
}

