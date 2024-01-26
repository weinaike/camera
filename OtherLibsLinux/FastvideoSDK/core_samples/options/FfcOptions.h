
#pragma once

#include "fastvideo_sdk_define.h"

class FfcOptions {
public:
	char* CompactedFfc;
	unsigned Divider;

	FfcOptions(void) { };
	~FfcOptions(void) { };

	bool Parse(int argc, char *argv[]);
};
