
#include "SelectChannelSampleOptions.h"

bool SelectChannelSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!SelectChannel.Parse(argc, argv)) {
		return false;
	}

	return true;
}
