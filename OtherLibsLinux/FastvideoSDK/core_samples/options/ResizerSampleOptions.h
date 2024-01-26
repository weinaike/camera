
#ifndef __RESIZER_SAMPLE_OPTIONS__
#define __RESIZER_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "ResizerOptions.h"

class ResizerSampleOptions : public virtual BaseOptions {
public:
	ResizerSampleOptions(void) {};
	~ResizerSampleOptions(void) {};

	ResizerOptions Resize;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __RESIZER_SAMPLE_OPTIONS__
