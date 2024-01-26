
#ifndef __NPP_RESIZE_SAMPLE_OPTIONS__
#define __NPP_RESIZE_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "NppResizeOptions.h"
#include "InterpolationOptions.h"

class NppResizeSampleOptions : public virtual BaseOptions {
public:
	NppResizeSampleOptions(void) {};
	~NppResizeSampleOptions(void) {};

	NppResizeOptions Resize;
	NppInterpolationOptions Interpolation;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __NPP_RESIZE_SAMPLE_OPTIONS__
