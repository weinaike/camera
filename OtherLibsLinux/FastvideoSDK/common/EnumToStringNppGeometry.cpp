#include "EnumToStringNppGeometry.h"

const char* EnumToString(fastNppGeometryTransformationType_t value) {
	switch (value) {
		case FAST_NPP_GEOMETRY_REMAP:
			return "FAST_NPP_GEOMETRY_REMAP";
		case FAST_NPP_GEOMETRY_REMAP3:
			return "FAST_NPP_GEOMETRY_REMAP3";
		case FAST_NPP_GEOMETRY_PERSPECTIVE:
			return "FAST_NPP_GEOMETRY_PERSPECTIVE";
		default:
			return "Other";
	}
}
