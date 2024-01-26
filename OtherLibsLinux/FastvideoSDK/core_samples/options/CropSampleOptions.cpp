
#include "CropSampleOptions.h"

bool CropSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Crop.Parse(argc, argv)) {
		return false;
	}

	return true;
}
