
#pragma once
#include "fastvideo_nppGeometry.h"
#include "ParametersParser.h"

class NppPerspectiveOptions {
public:
	NppPerspectiveOptions(void) {};
	~NppPerspectiveOptions(void) {};

	double Coeffs[3][3];
	bool Parse(int argc, char *argv[]);
};
