
#include "BinningOptions.h"
#include "ParametersParser.h"

#include <cstdio>
#include <cstring>

fastBinningMode_t BinningOptions::GetBinningModeFromString(const char *pattern) {
	fastBinningMode_t ret = FAST_BINNING_NONE;

	if (pattern != NULL) {
		if (strcmp(pattern, "sum") == 0)
			ret = FAST_BINNING_SUM;
		else if (strcmp(pattern, "avg") == 0)
			ret = FAST_BINNING_AVERAGE;
		else {
			fprintf(stderr, "Pattern %s was not recognized.\nSet to none\n", pattern);
			ret = FAST_BINNING_NONE;
		}
	}

	return ret;
}

bool BinningOptions::Parse(int argc, char *argv[]) {
	char *tmp = NULL;
	ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), "mode", &tmp);
	Mode = GetBinningModeFromString(tmp);

	Factor = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "factor");
	if (Factor < 1 || Factor > 4) {
		fprintf(stderr, "Unsupported factor size. Set to default\n");
		Factor = 1;
	}

	return true;
}
