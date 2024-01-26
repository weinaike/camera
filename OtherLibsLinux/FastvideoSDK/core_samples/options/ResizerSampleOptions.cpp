
#include "ResizerSampleOptions.h"

#include "ParametersParser.h"

bool ResizerSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Resize.Parse(argc, argv)) {
		return false;
	}

	return true;
}
