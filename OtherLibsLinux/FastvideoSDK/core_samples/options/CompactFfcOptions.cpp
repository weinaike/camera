
#include "CompactFfcOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <string>
#include <string.h>

bool CompactFfcOptions::Parse(int argc, char *argv[]) {
	DisableFiltering = ParametersParser::GetCmdLineArgumentInt(argc, (const char**)argv, "disableFiltering");
	return true;
}
