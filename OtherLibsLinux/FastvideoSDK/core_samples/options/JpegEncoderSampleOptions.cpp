
#include "JpegEncoderSampleOptions.h"

#include "ParametersParser.h"

bool JpegEncoderSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!JpegEncoder.Parse(argc, argv)) {
		return false;
	}

	return true;
}
