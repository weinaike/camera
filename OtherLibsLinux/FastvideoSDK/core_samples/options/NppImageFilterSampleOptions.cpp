
#include "NppImageFilterSampleOptions.h"

#include "ParametersParser.h"

bool NppImageFilterSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!NppImageFilter.Parse(argc, argv)) {
		return false;
	}

	return true;
}
