
#ifndef __SELECT_CHANNEL_OPTIONS__
#define __SELECT_CHANNEL_OPTIONS__

#include "fastvideo_sdk_define.h"

class SelectChannelOptions {
private:
	static fastChannelType_t GetChannelType(const char *str);

public:
	fastChannelType_t Channel;

	SelectChannelOptions(void) {};
	~SelectChannelOptions(void) {};

	bool Parse(int argc, char *argv[]);
};

#endif // __SELECT_CHANNEL_OPTIONS__
