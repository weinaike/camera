
#include "BaseColorCorrectionSampleOptions.h"

bool BaseColorCorrectionSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!BaseColorCorrection.Parse(argc, argv)) {
		return false;
	}

	return true;
}
