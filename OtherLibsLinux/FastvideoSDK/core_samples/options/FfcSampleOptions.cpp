
#include "FfcSampleOptions.h"

bool FfcSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Ffc.Parse(argc, argv)) {
		return false;
	}

	return true;
}
