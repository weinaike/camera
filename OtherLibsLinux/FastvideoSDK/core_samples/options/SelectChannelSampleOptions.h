
#ifndef __SELECT_CHANNEL_SAMPLE_OPTIONS__
#define __SELECT_CHANNEL_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "SelectChannelOptions.h"

class SelectChannelSampleOptions : public virtual BaseOptions {
public:
	SelectChannelSampleOptions(void) {};
	~SelectChannelSampleOptions(void) {};

	SelectChannelOptions SelectChannel;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __SELECT_CHANNEL_SAMPLE_OPTIONS__
