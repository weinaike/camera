
#ifndef __JPEG_DECODER_SAMPLE_OPTIONS__
#define __JPEG_DECODER_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "JpegDecoderOptions.h"

class JpegDecoderSampleOptions : public BaseOptions {
public:
	JpegDecoderSampleOptions(void) {};
	~JpegDecoderSampleOptions(void) {};

	JpegDecoderOptions JpegDecoder;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __JPEG_DECODER_SAMPLE_OPTIONS__
