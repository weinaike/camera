
#ifndef __SDI_CONVERTER_SAMPLE_OPTIONS__
#define __SDI_CONVERTER_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "SDIConverterOptions.h"

class SDIConverterSampleOptions : public virtual BaseOptions {
public:
	SDIConverterSampleOptions(void) {};
	~SDIConverterSampleOptions(void) {};

	SDIConverterOptions SDI;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __SDI_CONVERTER_SAMPLE_OPTIONS__
