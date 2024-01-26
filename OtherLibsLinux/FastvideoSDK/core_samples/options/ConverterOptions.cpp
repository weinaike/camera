
#include "ConverterOptions.h"
#include "ParametersParser.h"

#include <cstring>

fastRawFormat_t ConverterOptions::GetRawFormat(const char* str) {
	if (strcmp(str, "ptg12") == 0) {
		return FAST_RAW_PTG12;
	}
	if (strcmp(str, "ximea12") == 0) {
		return FAST_RAW_XIMEA12;
	}

	return FAST_RAW_XIMEA12;
}


bool ConverterOptions::Parse(int argc, char *argv[]) {
	Shift = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "shift");
	Randomize = ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "randomize");

	RawFormat = FAST_RAW_XIMEA12;
	char *tmp = NULL;
	if (ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), "format", &tmp)) {
		RawFormat = GetRawFormat(tmp);
	}

	return true;
}
