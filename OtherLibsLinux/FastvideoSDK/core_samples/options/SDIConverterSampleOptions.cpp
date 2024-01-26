
#include "SDIConverterSampleOptions.h"

bool SDIConverterSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!SDI.Parse(argc, argv)) {
		return false;
	}

	return true;
}
