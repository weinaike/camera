
#ifndef __LUT_SAMPLE_OPTIONS__
#define __LUT_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "LutOptions.h"

class LutSampleOptions : public virtual BaseOptions {
public:
	LutSampleOptions(void) {};
	~LutSampleOptions(void) {};

	LutOptions Lut;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __LUT_SAMPLE_OPTIONS__
