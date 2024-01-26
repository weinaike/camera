#ifndef __HELPER_PPM__
#define __HELPER_PPM__

#include "fastvideo_sdk_define.h"
#include "ppm.h"
#include "SurfaceTraits.hpp"
#include "IdentifySurface.hpp"
#include <memory>

template <class T, class Allocator> fastStatus_t
fvLoadPPM(
	const char *file, std::unique_ptr<T, Allocator> &data, 
	unsigned int &w, unsigned int &wPitch, unsigned int &h, 
	fastSurfaceFormat_t &surfaceFmt
) {
	Allocator alloc;
	void *p = NULL;
	unsigned channels = 0, bitsPerChannel = 0;
	if (!LoadPPM(file, &p, &alloc, w, wPitch, h, bitsPerChannel, channels))
		return FAST_IO_ERROR;
	data.reset((T*)p);
	surfaceFmt = IdentifySurface(bitsPerChannel, channels);
	return FAST_OK;
}

template <class T, class Allocator> fastStatus_t
fvSavePPM(const char *file, std::unique_ptr<T, Allocator> &data, fastSurfaceFormat_t surfaceFmt, unsigned int w, unsigned wPitch, unsigned int h) {
	unsigned char *p = (unsigned char *)data.get();
	const int bitsPerChannel = GetBitsPerChannelFromSurface(surfaceFmt);
	const int nChannels = GetNumberOfChannelsFromSurface(surfaceFmt);

	if (!SavePPM(file, (unsigned char *)data.get(), w, wPitch, h, bitsPerChannel, nChannels))
		return FAST_IO_ERROR;
	return FAST_OK;
}

#endif //
