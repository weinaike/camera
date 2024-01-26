
#ifndef __IMAGE_FILTER_SAMPLE_OPTIONS__
#define __IMAGE_FILTER_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "ImageFilterOptions.h"

class ImageFilterSampleOptions : public virtual BaseOptions {
public:
	ImageFilterOptions ImageFilter;

	ImageFilterSampleOptions(void) {};
	~ImageFilterSampleOptions(void) {};

	bool Parse(int argc, char *argv[]);
};

#endif // __IMAGE_FILTER_SAMPLE_OPTIONS__
