#include <exception>
#include <cstring>

#include "FastAllocator.h"
#include "fastvideo_sdk_define.h"

void* FastAllocator::allocate(size_t bytesCount) {
	void* p = NULL;
	if (fastMalloc((void**)&p, bytesCount) != FAST_OK) {
		throw std::bad_alloc();
	}
	return p;
}

void FastAllocator::deallocate(void* p) {
	if (fastFree(p) != FAST_OK) {
		throw std::bad_alloc();
	}
}

void FastAllocator::operator()(void* p) {
	if (fastFree(p) != FAST_OK) {
		throw std::bad_alloc();
	}
}

void FastAllocator::Copy(unsigned char* dst, unsigned char* src, const size_t count) {
	memcpy(dst, src, count);
}