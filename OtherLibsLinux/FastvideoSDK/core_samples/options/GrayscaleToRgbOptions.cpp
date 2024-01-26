
#include "GrayscaleToRgbOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <string>
#include <string.h>

bool GrayscaleToRgbOptions::Parse(int argc, char *argv[]) {
	ParametersParser::GetCmdLineArgumentString(argc, (const char**)argv, "iG", &ImageG);
	ParametersParser::GetCmdLineArgumentString(argc, (const char**)argv, "iB", &ImageB);
	if (ImageG == NULL || ImageB == NULL) {
		fprintf(stderr, "-iG or -iB parameter was not found\n");
		return false;
	}

	return true;
}
