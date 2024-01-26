
#include "ImageFilterSampleOptions.h"

#include "ParametersParser.h"

bool ImageFilterSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!ImageFilter.Parse(argc, argv)) {
		return false;
	}

	return true;
}
