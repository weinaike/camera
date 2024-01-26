
#include "LutSampleOptions.h"

bool LutSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Lut.Parse(argc, argv)) {
		return false;
	}

	return true;
}
