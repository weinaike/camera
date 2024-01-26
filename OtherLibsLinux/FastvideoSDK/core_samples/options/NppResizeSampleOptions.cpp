
#include "NppResizeSampleOptions.h"

bool NppResizeSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Resize.Parse(argc, argv)) {
		return false;
	}

	if (!Interpolation.Parse(argc, argv)) {
		return false;
	}
	return true;
}
