
#ifndef __JPEG_ENCODER_SAMPLE_OPTIONS__
#define __JPEG_ENCODER_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "JpegEncoderOptions.h"

class JpegEncoderSampleOptions : public BaseOptions {
public:
	JpegEncoderSampleOptions(void) {};
	~JpegEncoderSampleOptions(void) {};

	JpegEncoderOptions JpegEncoder;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __JPEG_ENCODER_SAMPLE_OPTIONS__
