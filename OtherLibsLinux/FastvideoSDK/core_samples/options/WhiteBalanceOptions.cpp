
#include "WhiteBalanceOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <string>
#include <string.h>

bool WhiteBalanceOptions::Parse(int argc, char *argv[]) {
	R = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "wb_r", 1.0));
	G1 = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "wb_g1", 1.0));
	G2 = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "wb_g2", 1.0));
	B = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "wb_b", 1.0));

	IsEnabled = R != 1.0f || G1 != 1.0f || G2 != 1.0f || B != 1.0f;

	return true;
}
