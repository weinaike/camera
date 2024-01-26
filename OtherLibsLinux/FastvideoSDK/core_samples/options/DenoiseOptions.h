
#ifndef __DENOISE_OPTIONS__
#define __DENOISE_OPTIONS__

#include "BaseOptions.h"
#include "fastvideo_sdk_define.h"

class DenoiseOptions : public virtual BaseOptions {
public:
	char *EnumToString, *FunctionName;
	fastWaveletType_t Wavelet;
	fastDenoiseThresholdFunctionType_t Function;
	float Threshold[3], Enhance[3];
	int DWT_Levels;

	long InputFilesize;

	DenoiseOptions() {};
	~DenoiseOptions() {};

	virtual bool Parse(int argc, char *argv[]);
};

#endif
