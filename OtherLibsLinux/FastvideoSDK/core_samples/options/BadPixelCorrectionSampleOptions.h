
#ifndef __BAD_PIXEL_CORRECTION_SAMPLE_OPTIONS__
#define __BAD_PIXEL_CORRECTION_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "BadPixelCorrectionOptions.h"
#include "DebayerOptions.h"

class BadPixelCorrectionSampleOptions : public virtual BaseOptions {
public:
	BadPixelCorrectionSampleOptions(void) {};
	~BadPixelCorrectionSampleOptions(void) {};

	BadPixelCorrectionOptions BadPixelCorrection;
	DebayerOptions Debayer;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __BAD_PIXEL_CORRECTION_SAMPLE_OPTIONS__
