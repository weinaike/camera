
#include "HistogramSampleOptions.h"

bool HistogramSampleOptions::Parse(int argc, char *argv[]) {
	if (!BaseOptions::Parse(argc, argv)) {
		return false;
	}

	if (!Histogram.Parse(argc, argv)) {
		return false;
	}

	return true;
}
