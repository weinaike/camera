
#include "GrayscaleToRgbSampleOptions.h"

bool GrayscaleToRgbSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!GrayscaleToRgb.Parse(argc, argv)) {
		return false;
	}

	return true;
}
