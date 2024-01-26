
#pragma once

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "FfcOptions.h"

class FfcSampleOptions : public virtual BaseOptions {
public:
	FfcSampleOptions(void) {};
	~FfcSampleOptions(void) {};

	FfcOptions Ffc;

	virtual bool Parse(int argc, char *argv[]);
};
