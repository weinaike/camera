
#ifndef __BIT_DEPTH_CONVERTER_OPTIONS__
#define __BIT_DEPTH_CONVERTER_OPTIONS__

#include "fastvideo_sdk_define.h"

class BitDepthConverterOptions {
public:
	unsigned BitsPerChannelDst;

	BitDepthConverterOptions(void) {};
	~BitDepthConverterOptions(void) {};

	bool Parse(int argc, char *argv[]);
};

#endif // __BIT_DEPTH_CONVERTER_OPTIONS__
