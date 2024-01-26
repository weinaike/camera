
#include "ToneCurveOptions.h"
#include "ParametersParser.h"

#include <cstdio>

bool ToneCurveOptions::Parse(int argc, char *argv[]) {
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "toneCurve", &ToneCurveFile);
	if (ToneCurveFile == NULL) {
		fprintf(stderr, "-toneCurve option was not found\n");
		return false;
	}

	return true;
}
