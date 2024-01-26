
#pragma once

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "GrayscaleToRgbOptions.h"

class GrayscaleToRgbSampleOptions : public virtual BaseOptions {
public:
	GrayscaleToRgbSampleOptions(void) {};
	~GrayscaleToRgbSampleOptions(void) {};

	GrayscaleToRgbOptions GrayscaleToRgb;

	virtual bool Parse(int argc, char *argv[]);
};
