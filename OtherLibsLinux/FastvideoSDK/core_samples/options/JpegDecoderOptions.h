
#ifndef __JPEG_DECODER_OPTIONS__
#define __JPEG_DECODER_OPTIONS__

#include "fastvideo_sdk_define.h"

class JpegDecoderOptions {
public:
	bool BayerCompression;

	JpegDecoderOptions(void) {
		BayerCompression = false;
	};
	~JpegDecoderOptions(void) { };

	bool Parse(int argc, char *argv[]);
};

#endif // __JPEG_DECODER_OPTIONS__
