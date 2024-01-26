
#pragma once

#include "BaseAllocator.h"

int LoadHeaderPPM(
	const char* fname,
	unsigned& width,
	unsigned& height,
	unsigned& numberOfChannels,
	unsigned& bitsPerChannel
);
int LoadPPM(
	const char* file,
	void** data,
	BaseAllocator* alloc,
	unsigned int& width,
	unsigned& wPitch,
	unsigned int& height,
	unsigned& bitsPerChannel,
	unsigned& channels
);
int SavePPM(
	const char *file,
	unsigned char *data,
	const unsigned w,
	const unsigned wPitch,
	const unsigned h,
	const int bitsPerChannel,
	const unsigned int channels
);
