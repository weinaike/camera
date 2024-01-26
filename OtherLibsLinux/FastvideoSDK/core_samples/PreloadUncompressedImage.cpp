
#include "PreloadUncompressedImage.hpp"

#include <list>
#include <string>

#include "helper_dir.hpp"
#include "bmp.hpp"
#include "ppm.h"
#include "tiff.h"
#include "checks.h"

fastStatus_t PreloadImageFromFile(
	const char* fname,
	unsigned& width,
	unsigned& height,
	unsigned& numberOfChannels,
	unsigned& bitsPerChannel
) {
	if (compareFileExtension(fname, ".pgm") ||
		compareFileExtension(fname, ".ppm")) {
		LoadHeaderPPM(fname, width, height, numberOfChannels, bitsPerChannel);
	} else if (compareFileExtension(fname, ".bmp")) {
		LoadHeaderBMP(fname, width, height, numberOfChannels, bitsPerChannel);
	} else if (compareFileExtension(fname, ".tif") ||
		compareFileExtension(fname, ".tiff")) {
		LoadHeaderTIFF(fname, width, height, numberOfChannels, bitsPerChannel);
	}

	return FAST_OK;
}

fastStatus_t PreloadImageFromFolder(
	const char* folderName,
	unsigned& width,
	unsigned& height,
	unsigned& numberOfChannels,
	unsigned& bitsPerChannel
) {
	std::list<std::string> files;
	{
		getFileList(folderName, files);
		if (files.empty()) {
			fprintf(stderr, "No input files found\n");
			return FAST_IO_ERROR;
		}
	}

	CHECK_FAST(PreloadImageFromFile(
		files.begin()->c_str(),
		width, height, numberOfChannels, bitsPerChannel
	));

	files.clear();
	return FAST_OK;
}


fastStatus_t PreloadImage(const char* path,
	bool isFolder,
	unsigned& width,
	unsigned& height,
	unsigned& numberOfChannels,
	unsigned& bitsPerChannel)
{
	return isFolder ? PreloadImageFromFolder(path, width, height, numberOfChannels, bitsPerChannel) : PreloadImageFromFile(path, width, height, numberOfChannels, bitsPerChannel);
}
