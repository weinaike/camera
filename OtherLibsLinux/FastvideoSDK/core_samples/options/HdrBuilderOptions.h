
#pragma once

#include "fastvideo_sdk_define.h"

class HdrBuilderOptions {
private:
	fastHDRBuilderFormat_t HdrFormatParser(char*);

public:
	fastHDRBuilderFormat_t Format;
	char* ExposureFileX1;
	char* ExposureFileX4;
	char* ExposureFileX16;

	HdrBuilderOptions(void) {
		ExposureFileX1 = ExposureFileX4 = ExposureFileX16 = nullptr;
	};
	~HdrBuilderOptions(void) { };

	bool Parse(int argc, char *argv[]);
};
