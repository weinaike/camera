#ifndef __JPEG_ENCODER_OPTIONS__
#define __JPEG_ENCODER_OPTIONS__

#include "fastvideo_sdk_define.h"

class JpegEncoderOptions {
private:
	fastJpegFormat_t ParseSubsamplingFmt(const int samplingFmtCode);

public:
	unsigned Quality;
	unsigned RestartInterval;
	fastJpegFormat_t SamplingFmt;

	bool GrayAsRGB;
	bool BayerCompression;

	bool Async;
	char *QuantTableFileName;
	bool noExif;

	JpegEncoderOptions(void) {
		BayerCompression = false;
	};
	~JpegEncoderOptions(void) { };

	bool Parse(int argc, char *argv[]);
};

#endif // __JPEG_ENCODER_OPTIONS__
