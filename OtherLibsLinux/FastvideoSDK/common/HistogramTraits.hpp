#ifndef __HISTOGRAM_TRAITS_H__
#define __HISTOGRAM_TRAITS_H__

#include "fastvideo_sdk_define.h"
#include "SurfaceTraits.hpp"

static int GetHistogramCount(
	fastHistogramType_t histogramType,
	fastSurfaceFormat_t surfaceFmt,
	const unsigned roiWidth,
	const unsigned minStride
) {
	switch (histogramType) {
		case FAST_HISTOGRAM_BAYER:
			return  3;
		case FAST_HISTOGRAM_BAYER_G1G2:
			return  4;
		case FAST_HISTOGRAM_PARADE:
			return 3 * uDivUp(roiWidth, minStride);
		default:
			return GetNumberOfChannelsFromSurface(surfaceFmt);
	}
}

#endif 

