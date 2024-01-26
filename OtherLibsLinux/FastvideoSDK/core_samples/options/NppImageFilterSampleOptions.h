
#ifndef __NPP_IMAGE_FILTER_SAMPLE_OPTIONS__
#define __NPP_IMAGE_FILTER_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "NppImageFilterOptions.h"

class NppImageFilterSampleOptions : public virtual BaseOptions {
public:
	NppImageFilterOptions NppImageFilter;

	NppImageFilterSampleOptions(void) {};
	~NppImageFilterSampleOptions(void) {};

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __NPP_IMAGE_FILTER_SAMPLE_OPTIONS__
