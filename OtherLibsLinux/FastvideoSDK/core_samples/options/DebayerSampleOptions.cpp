
#include "DebayerSampleOptions.h"

bool DebayerSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Debayer.Parse(argc, argv)) {
		return false;
	}

	if (!GrayscaleCorrection.Parse(argc, argv)) {
		return false;
	}

	if (!WhiteBalance.Parse(argc, argv)) {
		return false;
	}

	return true;
}
