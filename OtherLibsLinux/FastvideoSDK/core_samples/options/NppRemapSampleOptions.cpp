
#include "NppRemapSampleOptions.h"

bool NppRemapSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Remap.Parse(argc, argv)) {
		return false;
	}

	if (!Interpolation.Parse(argc, argv)) {
		return false;
	}
	return true;
}
