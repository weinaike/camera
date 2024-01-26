
#include "NppRotateOptions.h"
#include "ParametersParser.h"
#include "stdio.h"

bool NppRotateOptions::Parse(int argc, char *argv[]) {
	Angle = ParametersParser::GetCmdLineArgumentFloat(argc, const_cast<const char **>(argv), "angle");
	if (Angle < 0 || Angle > 360) {
		fprintf(stderr, "Incorrect angle in degree\n");
		return false;
	}
	Shift = ParametersParser::GetCmdLineArgumentFloat(argc, const_cast<const char **>(argv), "shift", 0);
	return true;
}
