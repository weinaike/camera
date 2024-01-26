
#ifndef __BINNING_SAMPLE_OPTIONS__
#define __BINNING_SAMPLE_OPTIONS__

#include "BaseOptions.h"
#include "BinningOptions.h"

class BinningSampleOptions : public virtual BaseOptions {
public:
	BinningSampleOptions(void) = default;
	~BinningSampleOptions(void) {}

	BinningOptions Binning;

	bool Parse(int argc, char *argv[]);
};

#endif // __BINNING_SAMPLE_OPTIONS__
