#include "EnumToStringCustomTransform.h"

const char *EnumToString(fastCustomTransformType_t type) {
	switch (type) {
		case FAST_TRANSFORM_NONE:
			return "none";
		case FAST_TRANSFORM_REARRANGE_2X8_TO_4X4:
			return "rearrange 2x8 to 4x4";
		case FAST_TRANSFORM_REARRANGE_2X4_TO_4X4_A:
			return "rearrange 2x4 to 4x4 (version A)";
		case FAST_TRANSFORM_REARRANGE_2X4_TO_4X4_B:
			return "rearrange 2x4 to 4x4 (version B)";
		case FAST_TRANSFORM_REARRANGE_MIX_2X2_AND_2X4_TO_4X2:
			return "rearrange mix 2x2 and 2x4 to 4x2";
		case FAST_TRANSFORM_SPLIT_BLOCK_BY_PIXEL:
			return "split block by pixel";
		case FAST_TRANSFORM_SPLIT_BLOCK_BY_PLANE:
			return "split block by plane";
		case FAST_TRANSFORM_COLORING_BLOCK:
			return "coloring block";
		case FAST_TRANSFORM_COLORING_PLANE:
			return "coloring plane";
		case FAST_TRANSFORM_BAYER_WHITE_BALANCE:
			return "white balance";
		default:
			return "unknown";
	}
}
