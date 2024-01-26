
#ifndef __BAD_PIXEL_CORRECTION_OPTIONS__
#define __BAD_PIXEL_CORRECTION_OPTIONS__

#include "fastvideo_sdk_define.h"

class BadPixelCorrectionOptions {
public:
	BadPixelCorrectionOptions(void) { };
	~BadPixelCorrectionOptions(void) { };

	bool Parse(int argc, char *argv[]);
};

#endif // __BAD_PIXEL_CORRECTION_OPTIONS__
