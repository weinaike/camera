
#ifndef __FFMPEG_OPTIONS__
#define __FFMPEG_OPTIONS__

#include "fastvideo_sdk_define.h"

class FfmpegOptions {
public:
	
	int FrameRepeat;
	double FrameRate;

	FfmpegOptions(void) {};
	~FfmpegOptions(void) {};

	bool Parse(int argc, char *argv[]);
};

#endif // __FFMPEG_OPTIONS__
