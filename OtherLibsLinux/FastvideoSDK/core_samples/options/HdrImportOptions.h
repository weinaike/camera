
#pragma once

#include "fastvideo_sdk_define.h"

class HdrImportOptions {
private:

public:
	unsigned LogOffset;

	HdrImportOptions(void) {
		LogOffset = 0;
	};
	~HdrImportOptions(void) { };

	bool Parse(int argc, char *argv[]);
};
