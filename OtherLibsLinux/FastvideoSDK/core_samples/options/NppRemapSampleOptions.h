
#ifndef __NPP_REMAP_SAMPLE_OPTIONS__
#define __NPP_REMAP_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "NppRemapOptions.h"
#include "InterpolationOptions.h"

class NppRemapSampleOptions : public virtual BaseOptions {
public:
	NppRemapSampleOptions(void) {};
	~NppRemapSampleOptions(void) {};

	NppRemapOptions Remap;
	NppInterpolationOptions Interpolation;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __NPP_REMAP_SAMPLE_OPTIONS__
