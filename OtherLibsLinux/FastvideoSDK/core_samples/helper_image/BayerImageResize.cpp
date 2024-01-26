
#include "fastvideo_sdk_define.h"

#include <cstdio>

#include "SurfaceTraits.hpp"
#include "FastAllocator.h"
#include "Image.h"

int BayerMergeLines(Image<FastAllocator> &img) {
	const unsigned srcWidth = img.GetBytesPerPixel() * img.w;

	const unsigned dstWidth = img.w * 2;
	const unsigned dstPitch = GetPitchFromSurface(img.surfaceFmt, dstWidth);
	const unsigned dstHeight = img.h / 2;

	if (srcWidth != img.wPitch) {
		FastAllocator alloc;
		unsigned char *tmp = (unsigned char*)alloc.allocate(dstPitch * dstHeight);

		for (unsigned srcY = 0, dstY = 0; srcY < img.h; srcY += 2, dstY++) {
			memcpy(
				&tmp           [dstY * dstPitch],
				&img.data.get()[srcY * img.wPitch],
				srcWidth
			);
			memcpy(
				&tmp           [ dstY      * dstPitch + srcWidth],
				&img.data.get()[(srcY + 1) * img.wPitch],
				srcWidth
			);
		}

		img.data.reset(tmp);
	}

	img.w = dstWidth;
	img.wPitch = dstPitch;
	img.h = dstHeight;
	return 0;
}

int BayerSplitLines(Image<FastAllocator> &img) {
	const unsigned srcWidth = img.GetBytesPerPixel() * img.w;

	const unsigned dstWidth = img.w / 2;
	const unsigned dstPitch = GetPitchFromSurface(img.surfaceFmt, dstWidth);
	const unsigned dstHeight = img.h * 2;
	const unsigned dstWidthInBytes = dstWidth * img.GetBytesPerPixel();

	if ((dstWidth * GetBytesPerChannelFromSurface(img.surfaceFmt)) != dstPitch) {
		FastAllocator alloc;
		unsigned char *tmp = (unsigned char*)alloc.allocate(dstPitch * dstHeight);

		for (unsigned srcY = 0, dstY = 0; srcY < img.h; srcY++, dstY += 2) {
			memcpy(
				&tmp[dstY * dstPitch],
				&img.data.get()[srcY * img.wPitch],
				dstWidthInBytes
			);
			memcpy(
				&tmp[(dstY + 1) * dstPitch],
				&img.data.get()[srcY * img.wPitch + dstWidthInBytes],
				dstWidthInBytes
			);
		}

		img.data.reset(tmp);
	}

	img.w = dstWidth;
	img.wPitch = dstPitch;
	img.h = dstHeight;
	return 0;
}
