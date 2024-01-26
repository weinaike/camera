
#ifndef __NPP_INTERPOLATION_OPTIONS__
#define __NPP_INTERPOLATION_OPTIONS__

#include "fastvideo_nppCommon.h"

class NppInterpolationOptions {

public:
	NppInterpolationOptions(void) {};
	~NppInterpolationOptions(void) {};

	fastNPPImageInterpolation_t Type;

	bool Parse(int argc, char *argv[]);
};

#endif // __NPP_INTERPOLATION_OPTIONS__
