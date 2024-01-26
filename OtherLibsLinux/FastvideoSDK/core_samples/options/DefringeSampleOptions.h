
#ifndef __DEFRINGE_SAMPLE_OPTIONS__
#define __DEFRINGE_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "DefringeOptions.h"

class DefringeSampleOptions : public virtual BaseOptions {
public:
	DefringeSampleOptions(void) {};
	~DefringeSampleOptions(void) {};

	DefringeOptions Defringe;

	bool Parse(int argc, char *argv[]);
};

#endif // __DEFRINGE_SAMPLE_OPTIONS__
