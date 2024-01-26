
#include "SelectChannelOptions.h"
#include "ParametersParser.h"

#include <cstring>

fastChannelType_t SelectChannelOptions::GetChannelType(const char* str) {
	if (strcmp(str, "R") == 0)
		return FAST_CHANNEL_R;
	if (strcmp(str, "G") == 0)
		return FAST_CHANNEL_G;
	if (strcmp(str, "B") == 0)
		return FAST_CHANNEL_B;
	return FAST_CHANNEL_R;
}


bool SelectChannelOptions::Parse(int argc, char *argv[]) {
	char *tmp;
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "channel", &tmp);

	Channel = GetChannelType(tmp);

	return true;
}
