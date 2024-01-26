
#include "MatrixGeneratorSampleOptions.h"

bool MatrixGeneratorSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Matrix.Parse(argc, argv)) {
		return false;
	}

	return true;
}
