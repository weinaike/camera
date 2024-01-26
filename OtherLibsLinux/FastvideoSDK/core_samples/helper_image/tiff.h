#pragma once

#include <stdint.h>

#include "BaseAllocator.h"

int LoadHeaderTIFF(
	const char* fname,
	unsigned& width,
	unsigned& height,
	unsigned& numberOfChannels,
	unsigned& bitsPerChannel
);

int LoadTIFF(
	const char* file, 
	void** data, 
	BaseAllocator* alloc, 
	unsigned& width, 
	unsigned& pitch, 
	unsigned& height, 
	unsigned& bitsPerChannel, 
	unsigned& channels
);

int SaveTIFF(
	const char* filename, 
	const void* _data, 
	uint32_t width, 
	uint32_t pitch, 
	uint32_t height, 
	uint32_t elsize,
	uint32_t channels
);
