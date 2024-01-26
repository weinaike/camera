
#pragma once

#include <stdio.h>

class BaseAllocator {
public:
	virtual void *allocate(size_t bytesCount) = 0;
	virtual void deallocate(void* p) = 0;
	virtual void operator()(void* p) = 0;

	static void Copy(unsigned char* dst, unsigned char* src, const size_t count);

	unsigned getAlignment();
};
