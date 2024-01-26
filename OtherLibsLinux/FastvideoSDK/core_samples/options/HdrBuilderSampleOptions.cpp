
#include "HdrBuilderSampleOptions.h"

bool HdrBuilderSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!HdrBuilder.Parse(argc, argv)) {
		return false;
	}

	return true;
}
