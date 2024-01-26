
#pragma once
#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "CropOptions.h"
#include "ResizerOptions.h"
#include "ImageFilterOptions.h"
#include "JpegEncoderOptions.h"

class Jpeg2JpegSampleOptions : public BaseOptions {
public:
	Jpeg2JpegSampleOptions(void) {};
	~Jpeg2JpegSampleOptions(void) {};

	CropOptions Crop;
	ResizerOptions Resize;
	ImageFilterOptions ImageFilter;
	JpegEncoderOptions JpegEncoder;
	
	virtual bool Parse(int argc, char *argv[]);
};
