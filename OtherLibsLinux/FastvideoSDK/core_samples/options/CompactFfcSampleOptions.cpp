
#include "CompactFfcSampleOptions.h"

bool CompactFfcSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!CompactFfc.Parse(argc, argv)) {
		return false;
	}

	return true;
}
