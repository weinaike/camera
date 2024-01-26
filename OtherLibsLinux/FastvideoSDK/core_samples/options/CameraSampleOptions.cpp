
#include "CameraSampleOptions.h"

bool CameraSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!JpegEncoder.Parse(argc, argv)) {
		return false;
	}

	if (!Debayer.Parse(argc, argv)) {
		return false;
	}

	if (!BaseColorCorrection.Parse(argc, argv)) {
		return false;
	}

	if (!GrayscaleCorrection.Parse(argc, argv)) {
		return false;
	}

	if (!FFMPEG.Parse(argc, argv)) {
		return false;
	}

	return true;
}
