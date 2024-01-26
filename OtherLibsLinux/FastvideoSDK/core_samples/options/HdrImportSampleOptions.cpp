
#include "HdrImportSampleOptions.h"

bool HdrImportSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!HdrImport.Parse(argc, argv)) {
		return false;
	}

	return true;
}
