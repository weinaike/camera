
#include "BinningSampleOptions.h"

bool BinningSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Binning.Parse(argc, argv)) {
		return false;
	}

	return true;
}
