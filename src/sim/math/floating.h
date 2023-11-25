#pragma once

#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884197
#endif

#ifdef DOUBLE_PRECISION
#define IS_FLOAT false
#define FLOAT double
#define SQRT sqrt
#define SIN sin
#define COS cos
#define TAN tan
#define ACOS acos
#define ASIN asin
#define ATAN2 atan2
#define COPYSIGN copysign
#define ABS abs
#ifdef __CUDACC__
#define UNIFORM_RANDOM curand_uniform_double
#endif
#else
#define IS_FLOAT true
#define FLOAT float
#define SQRT sqrtf
#define SIN sinf
#define COS cosf
#define TAN tanf
#define ACOS acosf
#define ASIN asinf
#define ATAN2 atan2f
#define COPYSIGN copysignf
#define ABS fabsf
#ifdef __CUDACC__
#define UNIFORM_RANDOM curand_uniform
#endif
#endif