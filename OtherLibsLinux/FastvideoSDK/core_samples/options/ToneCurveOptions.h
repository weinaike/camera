
#ifndef __TONE_CURVE_OPTIONS__
#define __TONE_CURVE_OPTIONS__

#include "fastvideo_sdk_define.h"

class ToneCurveOptions {
public:
	char *ToneCurveFile;

	ToneCurveOptions(void) {
		ToneCurveFile = NULL;
	};
	~ToneCurveOptions(void) {};

	bool Parse(int argc, char *argv[]);
};

#endif // __TONE_CURVE_OPTIONS__
