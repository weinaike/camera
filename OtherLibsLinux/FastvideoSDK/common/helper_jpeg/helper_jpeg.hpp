#ifndef __HELPER_JPEG__
#define __HELPER_JPEG__

#include "fastvideo_sdk_define.h"
#include <fstream>
#include <istream>
#include "JpegCommonDefines.hpp"
#include "Image.h"
#include "timing.hpp"

#define BYTE_SIZE 8U
#define DCT_SIZE 8U
#define MAX_CODE_LEN 16U

fastStatus_t jfifLoadHeader(
	fastJfifInfo_t *jfifInfo,
	std::istream &fd
);

fastStatus_t fastJfifHeaderLoadFromFile(const char *filename, fastJfifInfo_t *jfifInfo);
fastStatus_t fastJfifBytestreamLoadFromFile(const char *filename, fastJfifInfo_t *jfifInfo);

fastStatus_t fastJfifHeaderLoadFromMemory(
	const unsigned char	*inputStream,
	unsigned inputStreamSize,

	fastJfifInfo_t *jfifInfo
);

fastStatus_t fastJfifLoadFromMemory(
	const unsigned char	*inputStream,
	unsigned inputStreamSize,

	fastJfifInfo_t *jfifInfo
);

fastStatus_t fastJfifBytestreamLoadFromMemory(
	const unsigned char	*inputStream,
	unsigned inputStreamSize,

	fastJfifInfo_t *jfifInfo
);

fastStatus_t DLL fastJfifStoreToFile(
	const char *filename,
	fastJfifInfo_t *jfifInfo
);

fastStatus_t DLL fastJfifStoreToMemory(
	unsigned char *outputStream,
	unsigned *outputStreamSize,

	fastJfifInfo_t *jfifInfo
);

unsigned FileRemainingLength(std::istream& fd);

template<class Allocator>
fastStatus_t fastJfifLoadFromFile(
	std::string fname,
	JfifInfo<Allocator>* img,
	bool isHeaderToBytestream = false,
	double* loadTime = nullptr
) {
	hostTimer_t timer = NULL;
	if (loadTime != nullptr)
		timer = hostTimerCreate();

	std::ifstream input(fname.c_str(), std::ifstream::binary);
	if (!input.is_open()) {
		fprintf(stderr, "Input image file %s has not been found!\n", fname.c_str());
		return FAST_IO_ERROR;
	}

	input.exceptions(std::ios::eofbit | std::ios::failbit | std::ios::badbit);

	fastStatus_t res = jfifLoadHeader(&img->info, input);
	if (res != FAST_OK)
		return res;

	if (isHeaderToBytestream) {
		input.seekg(input.beg);
	}

	//Get length of the remainder
	unsigned streamSize = FileRemainingLength(input);

	Allocator alloc;
	if (loadTime != nullptr)
		hostTimerStart(timer);
	img->bytestream.reset((unsigned char*)alloc.allocate(streamSize));

	if (loadTime != nullptr)
		*loadTime = hostTimerEnd(timer);

	img->info.bytestreamSize = streamSize;

	//Burst read of the entropy-coded segment
	input.read((char*)(img->bytestream.get()), streamSize);
	input.close();

	if (loadTime != nullptr)
		hostTimerDestroy(timer);
	return FAST_OK;
}

void fastJfifFreeExif(fastJfifInfo_t* jfifInfo);

fastStatus_t PreloadJpegFromFolder(const char* folderName, fastJfifInfo_t* jfifInfo);


#endif // __HELPER_JPEG__
