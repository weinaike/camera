
#ifndef __MATRIX_GENERATOR_OPTIONS__
#define __MATRIX_GENERATOR_OPTIONS__

#include "fastvideo_sdk_define.h"

class MatrixGeneratorOptions {
public:
	float PixelValue;

	MatrixGeneratorOptions(void) { };
	~MatrixGeneratorOptions(void) { };

	bool Parse(int argc, char *argv[]);
};

#endif // __MATRIX_GENERATOR_OPTIONS__
