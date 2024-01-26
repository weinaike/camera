
#ifndef __DEBAYER_SAMPLE_OPTIONS__
#define __DEBAYER_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "DebayerOptions.h"
#include "GrayscaleCorrectionOptions.h"
#include "WhiteBalanceOptions.h"

class DebayerSampleOptions : public BaseOptions {
public:
	DebayerSampleOptions(void) { };
	~DebayerSampleOptions(void) {};

	DebayerOptions Debayer;
	GrayscaleCorrectionOptions GrayscaleCorrection;
	WhiteBalanceOptions WhiteBalance;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __DEBAYER_SAMPLE_OPTIONS__
