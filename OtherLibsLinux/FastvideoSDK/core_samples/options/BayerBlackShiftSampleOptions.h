
#ifndef __BLACK_SHIFT_SAMPLE_OPTIONS__
#define __BLACK_SHIFT_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "DebayerOptions.h"
#include "BayerBlackShiftOptions.h"

class BayerBlackShiftSampleOptions : public virtual BaseOptions {
public:
	BayerBlackShiftSampleOptions(void) {};
	~BayerBlackShiftSampleOptions(void) {};

	DebayerOptions Debayer;
	BayerBlackShiftOptions BayerBlackShift;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __BLACK_SHIFT_SAMPLE_OPTIONS__
