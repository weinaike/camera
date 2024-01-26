
#ifndef __CAMERA_SAMPLE_OPTIONS__
#define __CAMERA_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "JpegEncoderOptions.h"
#include "DebayerOptions.h"
#include "BaseColorCorrectionOptions.h"
#include "GrayscaleCorrectionOptions.h"
#include "FfmpegOptions.h"

class CameraSampleOptions : public virtual BaseOptions {
public:
	CameraSampleOptions(void) {};
	~CameraSampleOptions(void) {};

	JpegEncoderOptions JpegEncoder;
	DebayerOptions Debayer;
	BaseColorCorrectionOptions BaseColorCorrection;
	GrayscaleCorrectionOptions GrayscaleCorrection;
	FfmpegOptions FFMPEG;

	bool Parse(int argc, char *argv[]);
};

#endif // __CAMERA_SAMPLE_OPTIONS__
