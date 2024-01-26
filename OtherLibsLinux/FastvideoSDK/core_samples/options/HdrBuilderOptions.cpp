
#include "HdrBuilderOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <string>
#include <string.h>

fastHDRBuilderFormat_t HdrBuilderOptions::HdrFormatParser(char* format) {
	if (format != nullptr) {
		if (strcmp(format, "gray_3x12") == 0) {
			return FAST_HDR_GRAY_3x12;
		}
	}
	return FAST_HDR_GRAY_3x12;
}

bool HdrBuilderOptions::Parse(int argc, char *argv[]) {
	char *tmp = NULL;
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "hdrFormat", &tmp);
	Format = HdrFormatParser(tmp);

	ParametersParser::GetCmdLineArgumentString(argc, (const char**)argv, "ix1", &ExposureFileX1);
	if (ExposureFileX1 == nullptr) {
		fprintf(stderr, "Parameter -ix1 not found");
		return false;
	}

	ParametersParser::GetCmdLineArgumentString(argc, (const char**)argv, "ix4", &ExposureFileX4);
	if (ExposureFileX4 == nullptr) {
		fprintf(stderr, "Parameter -ix4 not found");
		return false;
	}

	ParametersParser::GetCmdLineArgumentString(argc, (const char**)argv, "ix16", &ExposureFileX16);
	if (ExposureFileX16 == nullptr) {
		fprintf(stderr, "Parameter -ix16 not found");
		return false;
	}

	return true;
}
