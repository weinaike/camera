#include <cuda_runtime.h>
#include <exception>
#include <cstdio>

#include "DevAllocator.h"

void* DevAllocator::allocate(size_t bytesCount) {
	void* p = NULL;
	if (cudaMalloc((void**)&p, bytesCount) != cudaSuccess) {
		throw std::bad_alloc();
	}
	return p;
}

void DevAllocator::deallocate(void* p) {
	if (cudaFree(p) != cudaSuccess) {
		throw std::bad_alloc();
	}
}

void DevAllocator::operator()(void* p) {
	if (cudaFree(p) != cudaSuccess) {
		throw std::bad_alloc();
	}
}

void DevAllocator::Copy(unsigned char* dst, unsigned char* src, const size_t count) {
	cudaError_t error = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
	if (error != cudaSuccess) {
		fprintf(stderr, "Memory copy error: %s\n", cudaGetErrorString(error));
	}
}