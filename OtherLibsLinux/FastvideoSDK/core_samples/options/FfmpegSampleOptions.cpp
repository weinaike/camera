
#include "FfmpegSampleOptions.h"

#include "ParametersParser.h"

bool FfmpegSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Ffmpeg.Parse(argc, argv)) {
		return false;
	}

	if (!JpegEncoder.Parse(argc, argv)) {
		return false;
	}

	return true;
}
