
#include "InterpolationOptions.h"
#include "ParametersParser.h"
#include "string.h"

fastNPPImageInterpolation_t ToInterpolationType(const char *str)  {
	if (strcmp(str, "linear") == 0) {
		return NPP_INTER_LINEAR;
	}
	else if (strcmp(str, "cubic") == 0) {
		return NPP_INTER_CUBIC;
	}
	else if (strcmp(str, "bspline") == 0) {
		return NPP_INTER_CUBIC2P_BSPLINE;
	}
	else if (strcmp(str, "catmullrom") == 0) {
		return NPP_INTER_CUBIC2P_CATMULLROM;
	}
	else if (strcmp(str, "b05c03") == 0) {
		return NPP_INTER_CUBIC2P_B05C03;
	}
	else if (strcmp(str, "super") == 0) {
		return NPP_INTER_SUPER;
	}
	else if (strcmp(str, "lanczos") == 0) {
		return NPP_INTER_LANCZOS;
	}
	return NPP_INTER_CUBIC;
}

bool NppInterpolationOptions::Parse(int argc, char *argv[]) {
	char *tmp = NULL;
	if (ParametersParser::GetCmdLineArgumentString(argc, const_cast<const char **>(argv), "interpolationType", &tmp)) {
		Type = ToInterpolationType(tmp);
	}
	else {
		Type = NPP_INTER_CUBIC;
	}

	return true;
}
