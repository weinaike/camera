
#ifndef __RESIZER_OPTIONS__
#define __RESIZER_OPTIONS__

#include "fastvideo_sdk_define.h"

class ResizerOptions {
public:
	static const unsigned SCALE_FACTOR_MAX = FAST_SCALE_FACTOR_MAX;
	static const unsigned MIN_SCALED_SIZE = FAST_MIN_SCALED_SIZE;

	unsigned OutputWidth;
	
	unsigned OutputHeight;
	bool OutputHeightEnabled;

	float ShiftX;
	float ShiftY;

	unsigned short Background[3];
	bool BackgroundEnabled;

	ResizerOptions() { }
	~ResizerOptions() { }

	bool Parse(int argc, char *argv[]);
};

#endif // __RESIZER_OPTIONS__
