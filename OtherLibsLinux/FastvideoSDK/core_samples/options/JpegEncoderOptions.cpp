#include "JpegEncoderOptions.h"
#include "ParametersParser.h"

#include <cstdio>

fastJpegFormat_t JpegEncoderOptions::ParseSubsamplingFmt(const int samplingFmtCode){
	fastJpegFormat_t samplingFmt = FAST_JPEG_444;

    if(samplingFmtCode == 444)
        samplingFmt = FAST_JPEG_444;
    else if(samplingFmtCode == 422)
        samplingFmt = FAST_JPEG_422;
    else if(samplingFmtCode == 420)
        samplingFmt = FAST_JPEG_420;
	else {
		fprintf(stderr, "Bad subsampling format, set to default...\n");
	}

	return samplingFmt;
}

bool JpegEncoderOptions::Parse(int argc, char *argv[]) {
	Quality = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "q");
	if (Quality <= 0 || Quality > 100) Quality = 75;

	RestartInterval = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "r");
	if (RestartInterval <= 0) RestartInterval = 16;

	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "quantTable", &QuantTableFileName);

	unsigned sub = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "s");
	if (sub <= 0) sub = 444;

	SamplingFmt = ParseSubsamplingFmt(sub);

	GrayAsRGB = ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "grayasrgb");
	BayerCompression = ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "bc");
	
	Async = ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "async");

	noExif = ParametersParser::CheckCmdLineFlag(argc, (const char **)argv, "noExif");

	return true;
}
