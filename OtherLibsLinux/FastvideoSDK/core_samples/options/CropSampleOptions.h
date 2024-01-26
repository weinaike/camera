
#ifndef __CROP_SAMPLE_OPTIONS__
#define __CROP_SAMPLE_OPTIONS__

#include "BaseOptions.h"
#include "CropOptions.h"

class CropSampleOptions : public virtual BaseOptions {
public:
	CropSampleOptions(void) = default;
	~CropSampleOptions(void) = default;

	CropOptions Crop;

	bool Parse(int argc, char *argv[]) override;
};

#endif // __CROP_SAMPLE_OPTIONS__
