
#ifndef __BAYER_BLACK_SHIFT_OPTIONS__
#define __BAYER_BLACK_SHIFT_OPTIONS__

#include "fastvideo_sdk_define.h"

class BayerBlackShiftOptions {
public:
	float BlackShift[3];

	BayerBlackShiftOptions(void) {};
	~BayerBlackShiftOptions(void) {};

	bool Parse(int argc, char *argv[]);
};

#endif // __BAYER_BLACK_SHIFT_OPTIONS__
