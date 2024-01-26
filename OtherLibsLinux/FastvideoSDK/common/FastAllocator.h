#pragma once

#include "BaseAllocator.h"

class FastAllocator : public BaseAllocator {
public:
	void* allocate(size_t bytesCount);
	void deallocate(void* p);
	void operator()(void* p);

	static void Copy(unsigned char* dst, unsigned char* src, const size_t count);
};
