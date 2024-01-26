
#include "DefringeSampleOptions.h"

bool DefringeSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Defringe.Parse(argc, argv)) {
		return false;
	}

	return true;
}
