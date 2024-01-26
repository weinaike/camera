
#ifndef __FFMPEG_SAMPLE_OPTIONS__
#define __FFMPEG_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "FfmpegOptions.h"
#include "JpegEncoderOptions.h"

class FfmpegSampleOptions : public virtual BaseOptions {
public:
	FfmpegSampleOptions(void) {};
	~FfmpegSampleOptions(void) {};

	FfmpegOptions Ffmpeg;
	JpegEncoderOptions JpegEncoder;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __FFMPEG_OPTIONS__
