#include "EnumToStringNppFilter.h"

const char* EnumToString(fastNPPImageFilterType_t value) {
	switch (value) {
		case NPP_GAUSSIAN_SHARPEN:
			return "NPP_GAUSSIAN_SHARPEN";
		case NPP_UNSHARP_MASK_SOFT:
			return "NPP_UNSHARP_MASK_SOFT";
		case NPP_UNSHARP_MASK_HARD:
			return "NPP_UNSHARP_MASK_HARD";
		default:
			return "Other";
	}
}
