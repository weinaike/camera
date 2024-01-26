
#include "LutDebayerSampleOptions.h"

bool LutDebayerSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Lut.Parse(argc, argv)) {
		return false;
	}
	if (!Debayer.Parse(argc, argv)) {
		return false;
	}

	return true;
}
