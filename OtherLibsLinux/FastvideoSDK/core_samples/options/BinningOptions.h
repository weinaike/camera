
#ifndef __BINNING_OPTIONS__
#define __BINNING_OPTIONS__

#include "fastvideo_sdk_define.h"

class BinningOptions {
	static fastBinningMode_t GetBinningModeFromString(const char *pattern);

public:
	fastBinningMode_t Mode;
	unsigned Factor;

	BinningOptions(void) { }
	~BinningOptions(void) = default;

	bool Parse(int argc, char *argv[]);
};

#endif // __BINNING_OPTIONS__
