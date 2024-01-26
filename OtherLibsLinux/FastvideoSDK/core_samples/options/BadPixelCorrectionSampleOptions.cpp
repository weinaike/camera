
#include "BadPixelCorrectionSampleOptions.h"

bool BadPixelCorrectionSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!BadPixelCorrection.Parse(argc, argv)) {
		return false;
	}

	if (!Debayer.Parse(argc, argv)) {
		return false;
	}
	return true;
}
