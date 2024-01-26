#include "DebayerJpegSampleOptions.h"

bool DebayerJpegSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!JpegEncoder.Parse(argc, argv)) {
		return false;
	}

	if (!Debayer.Parse(argc, argv)) {
		return false;
	}

	return true;
}
