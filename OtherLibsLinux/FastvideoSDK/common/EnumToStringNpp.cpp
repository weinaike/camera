#include "EnumToStringNpp.h"

const char* EnumToString(fastNPPImageInterpolation_t value) {
	switch (value) {
		case NPP_INTER_LINEAR:
			return "NPP_INTER_LINEAR";
		case NPP_INTER_CUBIC:
			return "NPP_INTER_CUBIC";
		case NPP_INTER_CUBIC2P_BSPLINE:
			return "NPP_INTER_CUBIC2P_BSPLINE";
		case NPP_INTER_CUBIC2P_CATMULLROM:
			return "NPP_INTER_CUBIC2P_CATMULLROM";
		case NPP_INTER_CUBIC2P_B05C03:
			return "NPP_INTER_CUBIC2P_B05C03";
		case NPP_INTER_SUPER:
			return "NPP_INTER_SUPER";
		case NPP_INTER_LANCZOS:
			return "NPP_INTER_LANCZOS";
		default:
			return "Other";
	}
}
