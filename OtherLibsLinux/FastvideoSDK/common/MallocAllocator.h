#pragma once
#include <exception>
#include "BaseAllocator.h"

class MallocAllocator : public BaseAllocator {
public:

	void *allocate(size_t bytesCount) {
		void* p = NULL;
		p = malloc(bytesCount);
		if (p == NULL) {
			throw std::bad_alloc();
		}
		return p;
	}

	void deallocate(void* p) {
		free(p);
	}

	void operator()(void* p) {
		free(p);
	}

	static void Copy(unsigned char* dst, unsigned char* src, const size_t count)
	{
		memcpy(dst, src, count);
	}
};
