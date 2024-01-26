
#include <cstdio>

#include "CropOptions.h"
#include "ParametersParser.h"

bool CropOptions::Parse(int argc, char *argv[]) {
	IsEnabled = true;

	char *tmp = NULL;
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, OptionName, &tmp);
	if (tmp != NULL) {
		// 375x375+63+0
		if (sscanf(tmp, "%d%*c%d%*c%d%*c%d", &CropWidth, &CropHeight, &CropLeftTopCoordsX, &CropLeftTopCoordsY) < 4) {
			fprintf(stderr, "Incorrect -%s option (-%s %s)\n", OptionName, OptionName, tmp);
			IsEnabled = false;
		}
	} else {
		IsEnabled = false;
	}

	return true;
}
