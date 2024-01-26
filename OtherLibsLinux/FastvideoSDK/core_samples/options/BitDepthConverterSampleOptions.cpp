
#include "BitDepthConverterSampleOptions.h"

bool BitDepthConverterSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!BitDepthConverter.Parse(argc, argv)) {
		return false;
	}

	return true;
}
