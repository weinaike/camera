
#ifndef __LUT_DEBAYER_SAMPLE_OPTIONS__
#define __LUT_DEBAYER_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "LutOptions.h"
#include "DebayerOptions.h"

class LutDebayerSampleOptions : public virtual BaseOptions {
public:
	LutDebayerSampleOptions(void) {};
	~LutDebayerSampleOptions(void) {};

	LutOptions Lut;
	DebayerOptions Debayer;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __LUT_DEBAYER_SAMPLE_OPTIONS__
