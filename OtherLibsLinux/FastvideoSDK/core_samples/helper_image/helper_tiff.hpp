#pragma once

#include "fastvideo_sdk_define.h"
#include "tiff.h"
#include "SurfaceTraits.hpp"
#include "IdentifySurface.hpp"
#include <memory>

template <class T, class Allocator> fastStatus_t
fvLoadTIFF(	const char* file, ImageT<T, Allocator>& image) {
	Allocator alloc;
	void* p = NULL;
	unsigned channels = 0, bitsPerChannel = 0;
		
	if (!LoadTIFF(file, &p, &alloc, image.w, image.wPitch, image.h, bitsPerChannel, channels))
		return FAST_IO_ERROR;
	image.data.reset((T*)p);

	image.surfaceFmt = IdentifySurface(bitsPerChannel, channels);

	return FAST_OK;
}

template <class T, class Allocator> fastStatus_t
fvSaveTIFF(const char* file, Image<Allocator>& image) {
	const int bytesPerChannel = GetBytesPerChannelFromSurface(image.surfaceFmt);
	const int nChannels = GetNumberOfChannelsFromSurface(image.surfaceFmt);

	if (!saveTIFF(file, (void*)image.data.get(), image.w, image.wPitch * sizeof(T), image.h, bytesPerChannel, nChannels))
		return FAST_IO_ERROR;
	return FAST_OK;
}

template <class T, class Allocator> fastStatus_t
fvSaveTIFF(const char* file, std::unique_ptr<T, Allocator>& data, fastSurfaceFormat_t surfaceFmt, unsigned int w, unsigned pitch, unsigned int h) {
	const int bytesPerChannel = GetBytesPerChannelFromSurface(surfaceFmt);
	const int nChannels = GetNumberOfChannelsFromSurface(surfaceFmt);

	if (!SaveTIFF(file, (void*)data.get(), w, pitch, h, bytesPerChannel, nChannels))
		return FAST_IO_ERROR;
	return FAST_OK;
}
