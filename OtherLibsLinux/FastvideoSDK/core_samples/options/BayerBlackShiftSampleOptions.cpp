
#include "BayerBlackShiftSampleOptions.h"

bool BayerBlackShiftSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Debayer.Parse(argc, argv)) {
		return false;
	}

	if (!BayerBlackShift.Parse(argc, argv)) {
		return false;
	}

	return true;
}
