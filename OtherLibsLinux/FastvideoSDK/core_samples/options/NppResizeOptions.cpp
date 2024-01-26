
#include "NppResizeOptions.h"
#include "ParametersParser.h"
#include "stdio.h"
#include "string.h"

bool NppResizeOptions::Parse(int argc, char *argv[]) {
	ResizedWidth = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "resizedWidth");
	if (ResizedWidth <= 0) {
		fprintf(stderr, "Incorrect resized image size\n");
		return false;
	}

	ResizedHeight = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "resizedHeight");
	ResizedHeightEnabled = ResizedHeight > 0;

	Shift = ParametersParser::GetCmdLineArgumentFloat(argc, const_cast<const char **>(argv), "shift", 0);

	return true;
}
