
#include "NppPerspectiveOptions.h"
#include <string>

bool NppPerspectiveOptions::Parse(int argc, char *argv[]) {
	std::string optionName("perspectiveCoeffs");
	

	if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), optionName.c_str())) {

		char *tmp = NULL;
		ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), optionName.c_str(), &tmp);

		if (sscanf(tmp,
			"%*c%lf%*c%lf%*c%lf%*c%lf%*c%lf%*c%lf%*c%lf%*c%lf%*c%lf%*c",
			&Coeffs[0][0],
			&Coeffs[0][1],
			&Coeffs[0][2],
			&Coeffs[1][0],
			&Coeffs[1][1],
			&Coeffs[1][2],
			&Coeffs[2][0],
			&Coeffs[2][1],
			&Coeffs[2][2]
			) < 9) {
			fprintf(stderr, "Incorrect -%s option (-%s %s)\n", optionName.c_str(), optionName.c_str(), tmp);
		}
	}
	else
	{
		Coeffs[0][0] = Coeffs[1][1] = Coeffs[2][2] = 1.;
		Coeffs[0][1] = Coeffs[0][2] = 0.;
		Coeffs[1][0] = Coeffs[1][2] = 0.;
		Coeffs[2][0] = Coeffs[2][1] = 0.;
	}
	return true;
}
