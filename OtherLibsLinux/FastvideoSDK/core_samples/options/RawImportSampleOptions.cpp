
#include "RawImportSampleOptions.h"

bool RawImportSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Raw.Parse(argc, argv)) {
		return false;
	}

	return true;
}
