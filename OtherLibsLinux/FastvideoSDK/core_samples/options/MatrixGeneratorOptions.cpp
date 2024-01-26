
#include "MatrixGeneratorOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <string>
#include <string.h>

bool MatrixGeneratorOptions::Parse(int argc, char *argv[]) {
	PixelValue = ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "pixel", 0.);
	return true;
}
