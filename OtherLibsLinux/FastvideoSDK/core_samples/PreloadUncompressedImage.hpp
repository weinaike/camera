
#pragma once
#include "fastvideo_sdk_define.h"

fastStatus_t PreloadImageFromFile(
	const char* fname,
	unsigned& width,
	unsigned& height,
	unsigned& numberOfChannels,
	unsigned& bitsPerChannel
);

fastStatus_t PreloadImageFromFolder(
	const char* folderName,
	unsigned &width,
	unsigned &height,
	unsigned &numberOfChannels,
	unsigned &bitsPerChannel
);


fastStatus_t PreloadImage(
	const char* folderName,
	bool isFolder,
	unsigned& width,
	unsigned& height,
	unsigned& numberOfChannels,
	unsigned& bitsPerChannel
);
