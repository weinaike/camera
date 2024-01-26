
#include "NppPerspectiveSampleOptions.h"
#include "NppPerspectiveOptions.h"

bool NppPerspectiveSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Interpolation.Parse(argc, argv)) {
		return false;
	}

	if (!Perspective.Parse(argc, argv)) {
		return false;
	}

	return true;
}
