
#ifndef __TONE_CURVE_SAMPLE_OPTIONS__
#define __TONE_CURVE_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "ToneCurveOptions.h"

class ToneCurveSampleOptions : public virtual BaseOptions {
public:
	ToneCurveSampleOptions(void) {};
	~ToneCurveSampleOptions(void) {};

	ToneCurveOptions ToneCurve;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __TONE_CURVE_SAMPLE_OPTIONS__
