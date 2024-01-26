
#include "NppRemapOptions.h"
#include "ParametersParser.h"

bool NppRemapOptions::Parse(int argc, char *argv[]) {
	Rotate90 = ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "rotate90");

	BackgroundEnabled = ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "R") &&
		ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "G") &&
		ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "B");
	if (BackgroundEnabled) {
		BackgroundR = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "R");
		BackgroundG = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "G");
		BackgroundB = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "B");
	}

	TransformType = ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "remap3") ?
		FAST_NPP_GEOMETRY_REMAP3 : FAST_NPP_GEOMETRY_REMAP;

	return true;
}
