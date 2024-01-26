
#ifndef __IMAGE_CONVERTER_SAMPLE_OPTIONS__
#define __IMAGE_CONVERTER_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "ConverterOptions.h"

class ImageConverterSampleOptions : public virtual BaseOptions {
public:
	ImageConverterSampleOptions(void) {};
	~ImageConverterSampleOptions(void) {};

	ConverterOptions ImageConverter;

	bool Parse(int argc, char *argv[]);
};

#endif // __IMAGE_CONVERTER_SAMPLE_OPTIONS__
