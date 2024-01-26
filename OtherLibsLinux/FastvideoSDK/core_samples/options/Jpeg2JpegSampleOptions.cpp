
#include "Jpeg2JpegSampleOptions.h"
#include "ParametersParser.h"

bool Jpeg2JpegSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Crop.Parse(argc, argv)) {
		return false;
	}

	if (!Resize.Parse(argc, argv)) {
		return false;
	}

	if (!ImageFilter.Parse(argc, argv)) {
		return false;
	}

	if (!JpegEncoder.Parse(argc, argv)) {
		return false;
	}

	return true;
}
