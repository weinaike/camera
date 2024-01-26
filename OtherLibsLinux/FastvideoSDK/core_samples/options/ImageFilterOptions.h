
#ifndef __IMAGE_FILTER_OPTIONS__
#define __IMAGE_FILTER_OPTIONS__

#include "fastvideo_sdk_define.h"

class ImageFilterOptions {
public:
	static double DisabledSharpConst;

	unsigned RawWidth;
	unsigned RawHeight;
	
	unsigned BitsCount;

	double SharpBefore;
	double SharpAfter;
	double Sigma;

	ImageFilterOptions(void) { };
	~ImageFilterOptions(void) { };

	bool Parse(int argc, char *argv[]);
};

#endif // __IMAGE_FILTER_OPTIONS__
