#ifndef __DEBAYER_OPTIONS__
#define __DEBAYER_OPTIONS__

#include "fastvideo_sdk_define.h"

class DebayerOptions {
private:
	static fastBayerPattern_t GetBayerPatternFromString(const char *pattern);
	static fastDebayerType_t GetBayerAlgorithmType(const char *pattern);

public:
	fastDebayerType_t BayerType;
	fastBayerPattern_t BayerFormat;

	DebayerOptions(void) { };
	~DebayerOptions(void) { };

	bool Parse(int argc, char *argv[]);
};

#endif // __DEBAYER_OPTIONS__
