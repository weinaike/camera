#ifndef __IMAGE_FILTER_VALIDATOR__
#define __IMAGE_FILTER_VALIDATOR__

#include "fastvideo_sdk_define.h"

#include <string>

typedef struct {
	fastImageFilterType_t imageFilterType;

	fastSurfaceFormat_t surfaces[3];
	std::string errorMessage;
} ImageFilterSurfaces_t;

#define FILTERS_COUNT 16
static const ImageFilterSurfaces_t ImageFilterSurfaces[FILTERS_COUNT] =
{
	{
		FAST_LUT_8_8,
		{
			FAST_I8,
			FAST_RGB8,
			FAST_BGR8
		},
		"Incorrect surface format (just RGB and Grayscale 8-bit)"
	},

	{
		FAST_LUT_8_8_C,
		{
			FAST_RGB8,
			FAST_BGR8,
			FAST_RGB8
		},
		"Incorrect surface format (just RGB 8-bit)"
	},

	{
		FAST_LUT_12_8,
		{
			FAST_RGB12,
			FAST_I12,
			FAST_RGB12
		},
		"Incorrect surface format (just RGB and Grayscale 12-bit)"
	},

	{
		FAST_LUT_12_8_C,
		{
			FAST_RGB12,
			FAST_RGB12,
			FAST_RGB12
		},
		"Incorrect surface format (just RGB 12-bit)"
	},

	{
		FAST_LUT_12_12,
		{
			FAST_RGB12,
			FAST_I12,
			FAST_RGB12
		},
		"Incorrect surface format (just RGB and Grayscale 12-bit)"
	},

	{
		FAST_LUT_12_12_C,
		{
			FAST_RGB12,
			FAST_RGB12,
			FAST_RGB12
		},
		"Incorrect surface format (just RGB 12-bit)"
	},

	{
		FAST_LUT_12_16,
		{
			FAST_RGB12,
			FAST_I12,
			FAST_RGB12
		},
		"Incorrect surface format (just RGB and Grayscale 12-bit)"
	},

	{
		FAST_LUT_12_16_C,
		{
			FAST_RGB12,
			FAST_RGB12,
			FAST_RGB12
		},
		"Incorrect surface format (just RGB 12-bit)"
	},

	{
		FAST_LUT_16_16,
		{
			FAST_RGB16,
			FAST_I16,
			FAST_RGB16
		},
		"Incorrect surface format (just RGB and Grayscale 16-bit)"
	},

	{
		FAST_LUT_16_16_C,
		{
			FAST_RGB16,
			FAST_RGB16,
			FAST_RGB16
		},
		"Incorrect surface format (just RGB 16-bit)"
	},

	{
		FAST_LUT_16_16_FR,
		{
			FAST_RGB16,
			FAST_I16,
			FAST_RGB16
		},
			"Incorrect surface format (just RGB and Grayscale 16-bit)"
	},

	{
		FAST_LUT_16_16_FR_C,
		{
			FAST_RGB16,
			FAST_RGB16,
			FAST_RGB16
		},
			"Incorrect surface format (just RGB 16-bit)"
	},

	{
		FAST_LUT_16_8,
		{
			FAST_RGB16,
			FAST_I16,
			FAST_RGB16
		},
		"Incorrect surface format (just RGB and Grayscale 16-bit)"
	},

	{
		FAST_LUT_16_8_C,
		{
			FAST_RGB16,
			FAST_RGB16,
			FAST_RGB16
		},
		"Incorrect surface format (just RGB 16-bit)"
	},

	{
		FAST_HSV_LUT_3D,
		{
			FAST_RGB12,
			FAST_RGB16,
			FAST_RGB16
		},
		"Incorrect surface format (just RGB 12/16-bit)"
	},

	{
		FAST_RGB_LUT_3D,
		{
			FAST_RGB12,
			FAST_RGB16,
			FAST_RGB16
		},
		"Incorrect surface format (just RGB 12/16-bit)"
	}
};

static bool ValidateSurface(fastImageFilterType_t imageFilter, fastSurfaceFormat_t surfaceFmt) {
	for (int i = 0; i < FILTERS_COUNT; i++) {
		if (ImageFilterSurfaces[i].imageFilterType == imageFilter) {
			for (int j = 0; j < 3; j++) {
				if (ImageFilterSurfaces[i].surfaces[j] == surfaceFmt) {
					return true;
				}
			}
			
			fprintf(stderr, "%s\n", ImageFilterSurfaces[i].errorMessage.c_str());
			return false;
		}
	}

	return false;
}

#endif // __IMAGE_FILTER_VALIDATOR__
