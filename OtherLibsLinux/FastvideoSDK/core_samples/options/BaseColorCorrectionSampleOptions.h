
#ifndef __BASE_COLOR_CORRECTION_SAMPLE_OPTIONS__
#define __BASE_COLOR_CORRECTION_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "BaseColorCorrectionOptions.h"

class BaseColorCorrectionSampleOptions : public virtual BaseOptions {
public:
	BaseColorCorrectionSampleOptions(void) {};
	~BaseColorCorrectionSampleOptions(void) {};

	BaseColorCorrectionOptions BaseColorCorrection;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __BASE_COLOR_CORRECTION_SAMPLE_OPTIONS__
