#ifndef __IDENTIFY_SURFACE__
#define __IDENTIFY_SURFACE__

#include "fastvideo_sdk_define.h"

static fastSurfaceFormat_t IdentifySurface(const unsigned bitsPerChannel, const unsigned nChannels) {
	if	(bitsPerChannel<= 8)
			return nChannels == 1 ? FAST_I8 : FAST_RGB8;
	if (bitsPerChannel <= 10 && nChannels == 1)
			return FAST_I10;
	if (bitsPerChannel <= 12)
			return nChannels == 1 ? FAST_I12 : FAST_RGB12;
	if (bitsPerChannel <= 14 && nChannels == 1)
			return FAST_I14;
	if (bitsPerChannel <= 16)
			return nChannels == 1 ? FAST_I16 : FAST_RGB16;

	return FAST_I8;
}

#endif //__SURFACE_TRAITS__
