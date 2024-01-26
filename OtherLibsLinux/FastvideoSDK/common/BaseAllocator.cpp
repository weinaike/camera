#include <exception>
#include <memory>
#include <stddef.h>

#include "BaseAllocator.h"

void BaseAllocator::Copy(unsigned char*, unsigned char*, const size_t) {
}

unsigned BaseAllocator::getAlignment() {
	return 4U;
}