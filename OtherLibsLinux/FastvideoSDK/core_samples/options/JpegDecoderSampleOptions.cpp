
#include "JpegDecoderSampleOptions.h"

#include "ParametersParser.h"

bool JpegDecoderSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!JpegDecoder.Parse(argc, argv)) {
		return false;
	}

	return true;
}
