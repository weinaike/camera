
#include "GrayscaleCorrectionOptions.h"
#include "ParametersParser.h"

#include <string>

bool GrayscaleCorrectionOptions::Parse(int argc, char *argv[]) {
	std::string matrixA("matrixA");
	std::string matrixB("matrixB");
	if (MadIndex != 0) {
		matrixA += "_" + std::to_string(MadIndex);
		matrixB += "_" + std::to_string(MadIndex);
	}

	ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), matrixA.c_str(), &MatrixA);
	ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), matrixB.c_str(), &MatrixB);
	return true;
}
