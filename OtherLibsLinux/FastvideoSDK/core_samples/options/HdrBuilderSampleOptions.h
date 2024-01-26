
#pragma once

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "HdrBuilderOptions.h"

class HdrBuilderSampleOptions : public virtual BaseOptions {
public:
	HdrBuilderSampleOptions(void) : BaseOptions(true, false, false) {
	}
	~HdrBuilderSampleOptions(void) {
	}

	HdrBuilderOptions HdrBuilder;

	virtual bool Parse(int argc, char *argv[]);
};
