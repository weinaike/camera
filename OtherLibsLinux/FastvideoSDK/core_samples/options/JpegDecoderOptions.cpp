
#include "JpegDecoderOptions.h"
#include "ParametersParser.h"

#include <cstdio>

bool JpegDecoderOptions::Parse(int argc, char *argv[]) {
	BayerCompression = ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "bc");
	return true;
}
