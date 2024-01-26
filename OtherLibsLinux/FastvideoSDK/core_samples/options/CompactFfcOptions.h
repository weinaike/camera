
#pragma once

#include "fastvideo_sdk_define.h"

class CompactFfcOptions {
public:
	bool DisableFiltering;

	CompactFfcOptions(void) { };
	~CompactFfcOptions(void) { };

	bool Parse(int argc, char *argv[]);
};
