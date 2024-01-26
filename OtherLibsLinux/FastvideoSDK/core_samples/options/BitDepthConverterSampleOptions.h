
#ifndef __BIT_DEPTH_CONVERTER_SAMPLE_OPTIONS__
#define __BIT_DEPTH_CONVERTER_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "BitDepthConverterOptions.h"

class BitDepthConverterSampleOptions : public virtual BaseOptions {
public:
	BitDepthConverterSampleOptions(void) {};
	~BitDepthConverterSampleOptions(void) {};

	BitDepthConverterOptions BitDepthConverter;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __BIT_DEPTH_CONVERTER_SAMPLE_OPTIONS__
