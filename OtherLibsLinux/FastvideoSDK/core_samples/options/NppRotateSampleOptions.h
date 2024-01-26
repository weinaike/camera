
#ifndef __NPP_ROTATE_SAMPLE_OPTIONS__
#define __NPP_ROTATE_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "NppRotateOptions.h"

class NppRotateSampleOptions : public virtual BaseOptions {
public:
	NppRotateSampleOptions(void) {};
	~NppRotateSampleOptions(void) {};

	NppRotateOptions Rotate;
	NppInterpolationOptions Interpolation;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __NPP_ROTATE_SAMPLE_OPTIONS__
