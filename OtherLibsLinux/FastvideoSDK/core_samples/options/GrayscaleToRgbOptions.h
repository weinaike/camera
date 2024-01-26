
#pragma once

#include "fastvideo_sdk_define.h"

class GrayscaleToRgbOptions {
public:
	char* ImageG;
	char* ImageB;

	GrayscaleToRgbOptions(void) { };
	~GrayscaleToRgbOptions(void) { };

	bool Parse(int argc, char *argv[]);
};
