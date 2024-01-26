
#ifndef __HELPER_YUV__
#define __HELPER_YUV__

#include <cstdlib>
#include <cstdio>

#include "fastvideo_sdk_define.h"

#include "helper_common.h"

template<typename T, class Allocator>
fastStatus_t loadYV12Planar(const char *fileName, std::unique_ptr<T, Allocator> &data, const long offset, const int size) {
	if (sizeof(T) != sizeof(unsigned char)) {
		return FAST_IO_ERROR;
	}

	FILE *fp = NULL;
	if (FOPEN_FAIL(FOPEN(fp, fileName, "rb")))
		return FAST_IO_ERROR;

	if (data == NULL) {
		Allocator alloc;
		CHECK_FAST_ALLOCATION(data.reset((T *)alloc.allocate(size)));
	}

	fseek(fp, offset, SEEK_SET);
	const int readSize = fread(data.get(), sizeof(unsigned char), size, fp);
	if (readSize != (sizeof(T) * size)) {
		fprintf(stderr, "Cannot read input stream\n");
		return FAST_INVALID_SIZE;
	}

	fclose(fp);
	return FAST_OK;
}

template<typename T, class Allocator>
fastStatus_t saveYV12Planar(const char *fileName, std::unique_ptr<T, Allocator> &data, const int size) {
	if (sizeof(T) != sizeof(unsigned char)) {
		return FAST_IO_ERROR;
	}

	FILE *fp = NULL;
	if (FOPEN_FAIL(FOPEN(fp, fileName, "ab")))
		return FAST_IO_ERROR;

	fseek(fp, 0, SEEK_END);

	const int readSize = fwrite(data.get(), sizeof(unsigned char), size, fp);
	if (readSize != (sizeof(T) * size)) {
		fprintf(stderr, "Cannot write output stream\n");
		return FAST_INVALID_SIZE;
	}

	fclose(fp);
	return FAST_OK;
}

#endif //__HELPER_YUV__
