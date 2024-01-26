
#include "BitDepthConverterOptions.h"
#include "ParametersParser.h"

#include <cstdio>

bool BitDepthConverterOptions::Parse(int argc, char *argv[]) {
	BitsPerChannelDst = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "dstBitsPerChannel");
	if (BitsPerChannelDst < 8 && BitsPerChannelDst > 16) {
		fprintf(stderr, "-dstBitsPerChannel parameter was not found or incorrent\n");
		return false;
	}

	return true;
}
