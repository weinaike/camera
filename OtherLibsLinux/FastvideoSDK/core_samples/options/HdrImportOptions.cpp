
#include <cstdio>
#include <cstring>

#include "HdrImportOptions.h"
#include "ParametersParser.h"

bool HdrImportOptions::Parse(int argc, char *argv[]) {
	char *tmp = NULL;

	LogOffset = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "logOffset");

	return true;
}
