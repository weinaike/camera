
#ifndef __CAMERA_MULTI_SAMPLE_OPTIONS__
#define __CAMERA_MULTI_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "JpegEncoderOptions.h"
#include "DebayerOptions.h"
#include "BaseColorCorrectionOptions.h"
#include "GrayscaleCorrectionOptions.h"
#include "FfmpegOptions.h"

class CameraMultiSampleOptions : public virtual BaseOptions {
public:
	char *OutputPath_2;

	JpegEncoderOptions JpegEncoder;
	DebayerOptions Debayer;
	BaseColorCorrectionOptions BaseColorCorrection_0;
	GrayscaleCorrectionOptions MAD_0;

	BaseColorCorrectionOptions BaseColorCorrection_1;
	GrayscaleCorrectionOptions MAD_1;
	char *Lut_1;

	FfmpegOptions FFMPEG;

	virtual bool Parse(int argc, char *argv[]);

	CameraMultiSampleOptions(void) : BaseColorCorrection_1(1), MAD_1(1) {};
	~CameraMultiSampleOptions(void) {};
};

#endif // __CAMERA_MULTI_SAMPLE_OPTIONS__
