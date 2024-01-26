
#include "AffineSampleOptions.h"

bool AffineSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Affine.Parse(argc, argv)) {
		return false;
	}

	return true;
}
