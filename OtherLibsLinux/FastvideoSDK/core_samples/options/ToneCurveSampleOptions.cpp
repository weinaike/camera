
#include "ToneCurveSampleOptions.h"

bool ToneCurveSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!ToneCurve.Parse(argc, argv)) {
		return false;
	}

	return true;
}
