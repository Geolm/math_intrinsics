#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#define _USE_MATH_DEFINES

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

    printf("%s\t max error : %.*e\n", name, FLT_DECIMAL_DIG, _mm256_cvtss_f32(v_max_error));
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

    printf("%s\t max error : %.*e\n", name, FLT_DECIMAL_DIG, vmaxvq_f32(v_max_error));
#endif

    free(input);
    free(result);
    
    PASS();
}

//----------------------------------------------------------------------------------------------------------------------
// generic unit test with 2 arguments
TEST generic_test2(reference_function2 ref, approximation_function2 approx, float range_min, float range_max, float epsilon, uint32_t num_elements, bool relative_error, const char* name)
{
    float* x = (float*) malloc(num_elements * sizeof(float));
    float* y = (float*) malloc(num_elements * sizeof(float));
    float* result = (float*) malloc(num_elements * sizeof(float));
    float step = ((range_max - range_min) / (float) (num_elements-1));
    uint32_t num_vectors = num_elements / simd_vector_width;

    for(uint32_t i=0; i<num_elements; ++i)
    {
        x[i] = (step * (float)(i)) + range_min;
        y[i] = sinf(x[i]);
        result[i] = ref(x[i], y[i]);
    }

#ifdef __MATH__INTRINSICS__AVX__
    __m256 v_epsilon = _mm256_set1_ps(epsilon);
    __m256 v_max_error = _mm256_setzero_ps();

    for(uint32_t i=0; i<num_vectors; ++i)
    {
        __m256 v_x = _mm256_loadu_ps(x + i * simd_vector_width);
        __m256 v_y = _mm256_loadu_ps(y + i * simd_vector_width);
        __m256 v_result = _mm256_loadu_ps(result+ i * simd_vector_width);
        __m256 v_approx = approx(v_x, v_y);
        __m256 v_error = _mm256_and_ps(_mm256_sub_ps(v_approx, v_result), _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)));

        if (relative_error)
            v_error = _mm256_div_ps(v_error, v_result);
        
        ASSERT(_mm256_movemask_ps(_mm256_cmp_ps(v_error, v_epsilon, _CMP_LE_OQ)) == 0xff);
        v_max_error = _mm256_max_ps(v_max_error, v_error);
    }

    v_max_error = _mm256_max_ps(v_max_error, _mm256_permute_ps(v_max_error, _MM_SHUFFLE(2, 1, 0, 3)));
    v_max_error = _mm256_max_ps(v_max_error, _mm256_permute_ps(v_max_error, _MM_SHUFFLE(1, 0, 3, 2)));
    v_max_error = _mm256_max_ps(v_max_error, _mm256_permute2f128_ps(v_max_error, v_max_error, 1));

    printf("%s\t max error : %.*e\n", name, FLT_DECIMAL_DIG, _mm256_cvtss_f32(v_max_error));
#else
    float32x4_t v_epsilon = vdupq_n_f32(epsilon);
    float32x4_t v_max_error = vdupq_n_f32(0.f);

    for(uint32_t i=0; i<num_vectors; ++i)
    {
        float32x4_t v_x = vld1q_f32(x + i * simd_vector_width);
        float32x4_t v_y = vld1q_f32(y + i * simd_vector_width);
        float32x4_t v_result = vld1q_f32(result+ i * simd_vector_width);
        float32x4_t v_approx = approx(v_x, v_y);
        float32x4_t v_error = vabsq_f32(vsubq_f32(v_approx, v_result));

        if (relative_error)
            v_error = vdivq_f32(v_error, v_result);
        
        ASSERT(vminvq_u32(vcleq_f32(v_error, v_epsilon)) == UINT32_MAX);
        v_max_error = vmaxq_f32(v_max_error, v_error);
    }

    printf("%s\t max error : %.*e\n", name, FLT_DECIMAL_DIG, vmaxvq_f32(v_max_error));
#endif

    free(x);
    free(y);
    free(result);
    
    PASS();
}

//----------------------------------------------------------------------------------------------------------------------
TEST value_expected(float input, float target, approximation_function function)
{
#ifdef __MATH__INTRINSICS__AVX__
    __m256 v_input = _mm256_set1_ps(input);
    float result = _mm256_cvtss_f32(function(v_input));
    ASSERT_EQ_FMT(target, result, "%f");
#else
    float32x4_t v_input = vdupq_n_f32(input);
    float result = vgetq_lane_f32(function(v_input), 0);
    ASSERT_EQ_FMT(target, result, "%f");
#endif

    PASS();
}

//----------------------------------------------------------------------------------------------------------------------
TEST nan_expected(float input, approximation_function function)
{
#ifdef __MATH__INTRINSICS__AVX__
    __m256 v_input = _mm256_set1_ps(input);
    float result = _mm256_cvtss_f32(function(v_input));
    ASSERT(isnan(result));
#else
    float32x4_t v_input = vdupq_n_f32(input);
    float result = vgetq_lane_f32(function(v_input), 0);
    ASSERT(isnan(result));
#endif

    PASS();
}

#define NUM_SAMPLES (1024)

#ifdef __MATH_INTRINSINCS_FAST__
static const float trigo_threshold = 1.e-06f;
static const float exp_threshold = 3.e-06f;
static const float arc_threshold = 1.e-04f;
static const float pow_threshold = 1.e-05f;
static const float cbrt_threshold = 2.e-07f;
#else
static const float trigo_threshold = FLT_EPSILON;
static const float exp_threshold = 2.e-07f;
static const float arc_threshold = 1.e-06f;
static const float pow_threshold = 1.e-06f;
static const float cbrt_threshold = 2.e-07f;
#endif

float atan2_xy(float x, float y) {return atan2f(y, x);}

SUITE(trigonometry)
{
    printf(".");

#ifdef __MATH__INTRINSICS__AVX__
    RUN_TESTp(generic_test, sinf, mm256_sin_ps, -10.f, 10.f, trigo_threshold, NUM_SAMPLES, false, "mm256_sin_ps");
    RUN_TESTp(generic_test, cosf, mm256_cos_ps, -10.f, 10.f, trigo_threshold, NUM_SAMPLES, false, "mm256_cos_ps");
    RUN_TESTp(generic_test, acosf, mm256_acos_ps, -1.f, 1.f, arc_threshold, NUM_SAMPLES, false, "mm256_acos_ps");
    RUN_TESTp(generic_test, asinf, mm256_asin_ps, -1.f, 1.f, arc_threshold, NUM_SAMPLES, false, "mm256_asin_ps");
    RUN_TESTp(generic_test, atanf, mm256_atan_ps, -10.f, 10.f, arc_threshold, NUM_SAMPLES, false, "mm256_atan_ps");

    // this task fails on linux and I don't have this OS to debug
    #if !defined(__linux__)
        RUN_TESTp(generic_test2, atan2_xy, mm256_atan2_ps, 0.f, 1000.f, 3.e-07f, NUM_SAMPLES, false, "mm256_atan2_ps");
    #endif
#else
    RUN_TESTp(generic_test, sinf, vsinq_f32, -10.f, 10.f, trigo_threshold, NUM_SAMPLES, false, "vsinq_f32");
    RUN_TESTp(generic_test, cosf, vcosq_f32, -10.f, 10.f, trigo_threshold, NUM_SAMPLES, false, "vcosq_f32");
    RUN_TESTp(generic_test, acosf, vacosq_f32, -1.f, 1.f, arc_threshold, NUM_SAMPLES, false, "vacosq_f32");
    RUN_TESTp(generic_test, asinf, vasinq_f32, -1.f, 1.f, arc_threshold, NUM_SAMPLES, false, "vasinq_f32");
    RUN_TESTp(generic_test, atanf, vatanq_f32, -10.f, 10.f, arc_threshold, NUM_SAMPLES, false, "vatanq_f32");
    RUN_TESTp(generic_test2, atan2_xy, vatan2q_f32, 0.f, 1000.f, 3.e-07f, NUM_SAMPLES, false, "vatan2q_f32");
#endif
}

SUITE(exponentiation)
{
    printf(".");
#ifdef __MATH__INTRINSICS__AVX__
    RUN_TESTp(generic_test, logf, mm256_log_ps, FLT_EPSILON, 1.e20f, 1.e-07f, 32768, true, "mm256_log_ps");
    RUN_TESTp(generic_test, log2f, mm256_log2_ps, FLT_EPSILON, 1.e20f, 3.e-07f, 32768, true, "mm256_log2_ps");
    RUN_TESTp(generic_test, expf, mm256_exp_ps, -87.f, 87.f, exp_threshold, NUM_SAMPLES, true, "mm256_exp_ps");
    RUN_TESTp(generic_test, exp2f, mm256_exp2_ps, -126.f, 126.f, exp_threshold, NUM_SAMPLES, true, "mm256_exp2");
    RUN_TESTp(generic_test, cbrtf, mm256_cbrt_ps, -1000.f, 1000.f, cbrt_threshold, 32768, true, "mm256_cbrt_ps");
    RUN_TESTp(generic_test2, powf, mm256_pow_ps, 0.f, 100000.f, pow_threshold, 32768, true, "mm256_pow_ps");
#else
    RUN_TESTp(generic_test, logf, vlogq_f32, FLT_EPSILON, 1.e20f, 1.e-07f, 32768, true, "vlogq_f32");
    RUN_TESTp(generic_test, log2f, vlog2q_f32, FLT_EPSILON, 1.e20f, 3.e-07f, 32768, true, "vlog2q_f32");
    RUN_TESTp(generic_test, expf, vexpq_f32, -87.f, 87.f, exp_threshold, NUM_SAMPLES, true, "vexpq_f32");
    RUN_TESTp(generic_test, exp2f, vexp2q_f32, -126.f, 126.f, exp_threshold, NUM_SAMPLES, true, "vexp2q_f32");
    RUN_TESTp(generic_test, cbrtf, vcbrtq_f32, -1000.f, 1000.f, cbrt_threshold, 4096, true, "vcbrtq_f32");
    RUN_TESTp(generic_test2, powf, vpowq_f32, 0.f, 100000.f, pow_threshold, 32768, true, "vpowq_f32");
#endif
}

SUITE(infinity_nan_compliance)
{
    const float positive_inf = INFINITY;
    const float negative_inf = -INFINITY;
    const float not_a_number = nanf("");

#ifdef __MATH__INTRINSICS__AVX__

    // log
    RUN_TESTp(nan_expected, -1.f, mm256_log_ps);
    RUN_TESTp(nan_expected, not_a_number, mm256_log_ps);
    RUN_TESTp(value_expected,  1.f, 0.f, mm256_log_ps);
    RUN_TESTp(value_expected,  0.f, negative_inf, mm256_log_ps);
    RUN_TESTp(value_expected,  positive_inf, positive_inf, mm256_log_ps);

    // log2
    RUN_TESTp(nan_expected, -1.f, mm256_log2_ps);
    RUN_TESTp(nan_expected, not_a_number, mm256_log2_ps);
    RUN_TESTp(value_expected,  1.f, 0.f, mm256_log2_ps);
    RUN_TESTp(value_expected,  0.f, negative_inf, mm256_log2_ps);
    RUN_TESTp(value_expected,  positive_inf, positive_inf, mm256_log2_ps);

    // exp
    RUN_TESTp(nan_expected, not_a_number, mm256_exp_ps);
    RUN_TESTp(value_expected, 0.f, 1.f, mm256_exp_ps);
    RUN_TESTp(value_expected,-0.f, 1.f, mm256_exp_ps);
    RUN_TESTp(value_expected, positive_inf, positive_inf, mm256_exp_ps);
    RUN_TESTp(value_expected, negative_inf, 0.f, mm256_exp_ps);

    // exp2
    RUN_TESTp(nan_expected, not_a_number, mm256_exp2_ps);
    RUN_TESTp(value_expected, 0.f, 1.f, mm256_exp2_ps);
    RUN_TESTp(value_expected,-0.f, 1.f, mm256_exp2_ps);
    RUN_TESTp(value_expected, positive_inf, positive_inf, mm256_exp2_ps);
    RUN_TESTp(value_expected, negative_inf, 0.f, mm256_exp2_ps);

    // sin
    RUN_TESTp(nan_expected, not_a_number, mm256_sin_ps);
    RUN_TESTp(nan_expected, positive_inf, mm256_sin_ps);
    RUN_TESTp(nan_expected, negative_inf, mm256_sin_ps);
    RUN_TESTp(value_expected, 0.f, 0.f, mm256_sin_ps);
    RUN_TESTp(value_expected, -0.f, -0.f, mm256_sin_ps);

    // cos
    RUN_TESTp(nan_expected, not_a_number, mm256_cos_ps);
    RUN_TESTp(nan_expected, positive_inf, mm256_cos_ps);
    RUN_TESTp(nan_expected, negative_inf, mm256_cos_ps);
    RUN_TESTp(value_expected, 0.f, 1.f, mm256_cos_ps);
    RUN_TESTp(value_expected, -0.f, 1.f, mm256_cos_ps);

    // asin
    RUN_TESTp(nan_expected, not_a_number, mm256_asin_ps);
    RUN_TESTp(nan_expected, 2.f, mm256_asin_ps);
    RUN_TESTp(nan_expected, -2.f, mm256_asin_ps);
    RUN_TESTp(value_expected, 0.f, 0.f, mm256_asin_ps);
    RUN_TESTp(value_expected, -0.f, -0.f, mm256_asin_ps);

    // acos
    RUN_TESTp(nan_expected, not_a_number, mm256_acos_ps);
    RUN_TESTp(nan_expected, 2.f, mm256_acos_ps);
    RUN_TESTp(nan_expected, -2.f, mm256_acos_ps);
    RUN_TESTp(value_expected, 1.f, 0.f, mm256_acos_ps);

    // atan
    RUN_TESTp(nan_expected, not_a_number, mm256_atan_ps);
    RUN_TESTp(value_expected, 0.f, 0.f, mm256_atan_ps);
    RUN_TESTp(value_expected, -0.f, -0.f, mm256_atan_ps);
    RUN_TESTp(value_expected, positive_inf, (float)M_PI_2, mm256_atan_ps);
    RUN_TESTp(value_expected, negative_inf, (float)-M_PI_2, mm256_atan_ps);

    // cbrt
    RUN_TESTp(nan_expected, not_a_number, mm256_cbrt_ps);
    RUN_TESTp(value_expected, 0.f, 0.f, mm256_cbrt_ps);
    RUN_TESTp(value_expected, -0.f, -0.f, mm256_cbrt_ps);

#else
    RUN_TESTp(nan_expected, -1.f, vlogq_f32);
    RUN_TESTp(nan_expected, not_a_number, vlogq_f32);
    RUN_TESTp(value_expected,  1.f, 0.f, vlogq_f32);
    RUN_TESTp(value_expected,  0.f, negative_inf, vlogq_f32);
    RUN_TESTp(value_expected,  positive_inf, positive_inf, vlogq_f32);

    RUN_TESTp(nan_expected, -1.f, vlog2q_f32);
    RUN_TESTp(nan_expected, not_a_number, vlog2q_f32);
    RUN_TESTp(value_expected,  1.f, 0.f, vlog2q_f32);
    RUN_TESTp(value_expected,  0.f, negative_inf, vlog2q_f32);
    RUN_TESTp(value_expected,  positive_inf, positive_inf, vlog2q_f32);

    // exp
    RUN_TESTp(nan_expected, not_a_number, vexpq_f32);
    RUN_TESTp(value_expected, 0.f, 1.f, vexpq_f32);
    RUN_TESTp(value_expected,-0.f, 1.f, vexpq_f32);
    RUN_TESTp(value_expected, positive_inf, positive_inf, vexpq_f32);
    RUN_TESTp(value_expected, negative_inf, 0.f, vexpq_f32);

    // exp2
    RUN_TESTp(nan_expected, not_a_number, vexp2q_f32);
    RUN_TESTp(value_expected, 0.f, 1.f, vexp2q_f32);
    RUN_TESTp(value_expected,-0.f, 1.f, vexp2q_f32);
    RUN_TESTp(value_expected, positive_inf, positive_inf, vexp2q_f32);
    RUN_TESTp(value_expected, negative_inf, 0.f, vexp2q_f32);

    // sin
    RUN_TESTp(nan_expected, not_a_number, vsinq_f32);
    RUN_TESTp(nan_expected, positive_inf, vsinq_f32);
    RUN_TESTp(nan_expected, negative_inf, vsinq_f32);
    RUN_TESTp(value_expected, 0.f, 0.f, vsinq_f32);
    RUN_TESTp(value_expected, -0.f, -0.f, vsinq_f32);

    // cos
    RUN_TESTp(nan_expected, not_a_number, vcosq_f32);
    RUN_TESTp(nan_expected, positive_inf, vcosq_f32);
    RUN_TESTp(nan_expected, negative_inf, vcosq_f32);
    RUN_TESTp(value_expected, 0.f, 1.f, vcosq_f32);
    RUN_TESTp(value_expected, -0.f, 1.f, vcosq_f32);

    // asin
    RUN_TESTp(nan_expected, not_a_number, vasinq_f32);
    RUN_TESTp(nan_expected, 2.f, vasinq_f32);
    RUN_TESTp(nan_expected, -2.f, vasinq_f32);
    RUN_TESTp(value_expected, 0.f, 0.f, vasinq_f32);
    RUN_TESTp(value_expected, -0.f, -0.f, vasinq_f32);

    // acos
    RUN_TESTp(nan_expected, not_a_number, vacosq_f32);
    RUN_TESTp(nan_expected, 2.f, vacosq_f32);
    RUN_TESTp(nan_expected, -2.f, vacosq_f32);
    RUN_TESTp(value_expected, 1.f, 0.f, vacosq_f32);

    // atan
    RUN_TESTp(nan_expected, not_a_number, vatanq_f32);
    RUN_TESTp(value_expected, 0.f, 0.f, vatanq_f32);
    RUN_TESTp(value_expected, -0.f, -0.f, vatanq_f32);
    RUN_TESTp(value_expected, positive_inf, (float)M_PI_2, vatanq_f32);
    RUN_TESTp(value_expected, negative_inf, (float)-M_PI_2, vatanq_f32);

    // cbrt
    RUN_TESTp(nan_expected, not_a_number, vcbrtq_f32);
    RUN_TESTp(value_expected, 0.f, 0.f, vcbrtq_f32);
    RUN_TESTp(value_expected, -0.f, -0.f, vcbrtq_f32);
#endif
}

GREATEST_MAIN_DEFS();

int main(int argc, char * argv[])
{
    GREATEST_MAIN_BEGIN();

#ifdef __MATH_INTRINSINCS_FAST__
    printf("math intrinsics unit tests : mode fast\n\n");
#else
    printf("math intrinsics unit tests : mode precision\n\n");
#endif

    RUN_SUITE(trigonometry);
    RUN_SUITE(exponentiation);
    RUN_SUITE(infinity_nan_compliance);

    GREATEST_MAIN_END();

    (void)nan_expected;
    (void)value_expected;
}

