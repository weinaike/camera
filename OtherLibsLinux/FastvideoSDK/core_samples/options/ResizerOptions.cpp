
#include "ResizerOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <cstring>

bool ResizerOptions::Parse(int argc, char *argv[]) {
	OutputWidth = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "outputWidth");
	if (argc > 1 && OutputWidth < MIN_SCALED_SIZE) {
		fprintf(stderr, "Unsupported output image width - %d. Minimum width is %d\n", OutputWidth, MIN_SCALED_SIZE);
		return false;
	}

	OutputHeight = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "outputHeight");
	OutputHeightEnabled = OutputHeight > 0;

	BackgroundEnabled = false;
	char *tmp = NULL;
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "background", &tmp);
	if (tmp != NULL) {
		// 375,375,63
		const int count = sscanf(tmp, "%d%*c%d%*c%d", &(Background[0]), &(Background[1]), &(Background[2]));
		if (count != 3 && count != 1) {
			fprintf(stderr, "Incorrect -background option (-background %s)\n", tmp);
		} else {
			BackgroundEnabled = true;
		}
	}

	ShiftX = 0.0f;
	ShiftY = 0.0f;

	return true;
}
