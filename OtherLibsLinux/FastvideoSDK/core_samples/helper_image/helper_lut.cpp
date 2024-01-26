#include <iostream>
#include <memory>

#include "fastvideo_sdk_define.h"
#include "FastAllocator.h"
#include "helper_common.h"
#include "cube.h"

template<class T, class Allocator> fastStatus_t
fvLoadLut(const char *file, std::unique_ptr<T, Allocator> &data, unsigned size) {
	FILE *fp = NULL;

	if (FOPEN_FAIL(FOPEN(fp, file, "r")))
		return FAST_IO_ERROR;

	Allocator alloc;
	data.reset((T*)alloc.allocate(size * sizeof(T)));

	unsigned i = 0;
	while (i < size && !feof(fp)) {
		float value;
		fscanf(fp, "%f", &value);
		data.get()[i] = static_cast<T>(value);
		i++;
	}

	if (i != size) {
		return FAST_IO_ERROR;
	}

	fclose(fp);
	return FAST_OK;
}

template fastStatus_t fvLoadLut(const char *file, std::unique_ptr<unsigned char, FastAllocator> &data, unsigned size);
template fastStatus_t fvLoadLut(const char *file, std::unique_ptr<unsigned short, FastAllocator> &data, unsigned size);
template fastStatus_t fvLoadLut(const char *file, std::unique_ptr<float, FastAllocator> &data, unsigned size);
