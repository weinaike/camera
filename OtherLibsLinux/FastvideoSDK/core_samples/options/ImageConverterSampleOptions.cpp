
#include "ImageConverterSampleOptions.h"

#include "ParametersParser.h"

bool ImageConverterSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!ImageConverter.Parse(argc, argv)) {
		return false;
	}

	return true;
}
