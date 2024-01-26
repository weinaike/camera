
#include "FfmpegOptions.h"
#include "ParametersParser.h"

bool FfmpegOptions::Parse(int argc, char *argv[]) {

	FrameRepeat = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "frameRepeat");
	if (FrameRepeat <= 0) {
		FrameRepeat = 1;
	}
	FrameRate = ParametersParser::GetCmdLineArgumentFloat(argc, (const char **)argv, "frameRate", 24.0);

	return true;
}
