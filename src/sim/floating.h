#pragma once

#ifdef DOUBLE_PRECISION
#define FLOAT double
#define SQRT sqrt
#else
#define FLOAT float
#define SQRT sqrtf
#endif