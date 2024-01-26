
#include "NppRotateSampleOptions.h"

bool NppRotateSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Rotate.Parse(argc, argv)) {
		return false;
	}

	if (!Interpolation.Parse(argc, argv)) {
		return false;
	}

	return true;
}
