
#include "CameraMultiSampleOptions.h"
#include "ParametersParser.h"

bool CameraMultiSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!JpegEncoder.Parse(argc, argv)) {
		return false;
	}

	if (!Debayer.Parse(argc, argv)) {
		return false;
	}

	if (!BaseColorCorrection_0.Parse(argc, argv)) {
		return false;
	}
	if (!BaseColorCorrection_1.Parse(argc, argv)) {
		return false;
	}

	if (!MAD_0.Parse(argc, argv)) {
		return false;
	}
	if (!MAD_1.Parse(argc, argv)) {
		return false;
	}

	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "o2", &OutputPath_2);
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "lut_1", &Lut_1);

	if (!FFMPEG.Parse(argc, argv)) {
		return false;
	}

	return true;
}
