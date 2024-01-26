
#include "FfcOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <string>
#include <string.h>

bool FfcOptions::Parse(int argc, char *argv[]) {
	ParametersParser::GetCmdLineArgumentString(argc, (const char**)argv, "compactedFfc", &CompactedFfc);
	if (CompactedFfc == NULL) {
		fprintf(stderr, "-compactedFfc parameter was not found\n");
		return false;
	}

	Divider = ParametersParser::GetCmdLineArgumentInt(argc, (const char**)argv, "divider");
	return true;
}
