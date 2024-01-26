
#ifndef __MATRIX_GENERATOR_SAMPLE_OPTIONS__
#define __MATRIX_GENERATOR_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "MatrixGeneratorOptions.h"

class MatrixGeneratorSampleOptions : public virtual BaseOptions {
public:
	MatrixGeneratorSampleOptions(void) {};
	~MatrixGeneratorSampleOptions(void) {};

	MatrixGeneratorOptions Matrix;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __MATRIX_GENERATOR_SAMPLE_OPTIONS__
