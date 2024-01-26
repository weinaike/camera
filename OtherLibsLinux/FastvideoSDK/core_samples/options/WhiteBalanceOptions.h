
#ifndef __WHITE_BALANCE_OPTIONS__
#define __WHITE_BALANCE_OPTIONS__

#include "BaseOptions.h"
#include "WhiteBalanceOptions.h"

#include "fastvideo_sdk_define.h"

class WhiteBalanceOptions {
public:
	float R;
	float G1;
	float G2;
	float B;

	bool IsEnabled;

	WhiteBalanceOptions(void) {};
	~WhiteBalanceOptions(void) {};

	bool Parse(int argc, char *argv[]);
};

#endif // __WHITE_BALANCE_OPTIONS__
