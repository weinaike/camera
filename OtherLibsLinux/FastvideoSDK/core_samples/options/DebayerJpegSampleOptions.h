
#ifndef __DEBAYER_JPEG_SAMPLE_OPTIONS__
#define __DEBAYER_JPEG_SAMPLE_OPTIONS__

#include "BaseOptions.h"
#include "DebayerOptions.h"
#include "JpegEncoderOptions.h"

class DebayerJpegSampleOptions : public virtual BaseOptions {
public:
	DebayerJpegSampleOptions(void) {};
	~DebayerJpegSampleOptions(void) {};

	JpegEncoderOptions JpegEncoder;
	DebayerOptions Debayer;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __DEBAYER_JPEG_SAMPLE_OPTIONS__
