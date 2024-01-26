
#include "BayerBlackShiftOptions.h"
#include "ParametersParser.h"

bool BayerBlackShiftOptions::Parse(int argc, char *argv[]) {
	BlackShift[0] = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "blackShiftR"));
	BlackShift[1] = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "blackShiftG"));
	BlackShift[2] = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "blackShiftB"));

	return true;
}
