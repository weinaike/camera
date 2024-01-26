
#ifndef __AFFINE_SAMPLE_OPTIONS__
#define __AFFINE_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "AffineOptions.h"

class AffineSampleOptions : public virtual BaseOptions {
public:
	AffineSampleOptions(void) {};
	~AffineSampleOptions(void) {};

	AffineOptions Affine;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __AFFINE_SAMPLE_OPTIONS__
