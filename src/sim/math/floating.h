#pragma once

#ifdef DOUBLE_PRECISION
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
#else
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
#endif