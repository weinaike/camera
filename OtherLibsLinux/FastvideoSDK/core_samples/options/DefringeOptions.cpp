
#include "DefringeOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <string>
#include <vector>
#include <sstream>

bool DefringeOptions::Parse(int argc, char *argv[]) {
	WindowSize = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "window");
	if (WindowSize <= 0 || WindowSize > 40) {
		fprintf(stderr, "Unsupported window size\n");
		return false;
	}

	TintR = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "tintR");
	TintG = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "tintG");
	TintB = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "tintB");

	Fi_tint = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "fi_tint"));
	if (Fi_tint == 0) {
		Fi_tint = -190;
	}

	Fi_max = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "fi_max"));
	if (Fi_max == 0) {
		Fi_max = 60;
	}

	Coefficient = static_cast<float>(ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "coefficient"));
	if (Coefficient == 0.f) {
		Coefficient = 0.1f;
	}

	return true;
}
