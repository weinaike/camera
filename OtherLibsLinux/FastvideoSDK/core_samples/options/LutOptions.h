
#ifndef __LUT_OPTIONS__
#define __LUT_OPTIONS__

#include "fastvideo_sdk_define.h"
#include "SampleTypes.h"

class LutOptions {
public:
	char *Lut;

	char *Lut_R;
	char *Lut_G;
	char *Lut_B;

	unsigned lutSize1D;
	fast_uint3 lutSize3D;

	fastColorSaturationOperationType_t OperationType[3];
	fastColorSaturationChannelType_t ChannelType[3];

	fastImageFilterType_t ImageFilter;

	LutOptions(void) {};
	~LutOptions(void) {};

	bool Parse(int argc, char *argv[]);
	bool Parse(int argc, char *argv[], const char *lut);
};

#endif // __LUT_OPTIONS__
