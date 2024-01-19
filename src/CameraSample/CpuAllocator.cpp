#include <exception>
#include <cstring>

#include <cuda_runtime.h>
#include <CpuAllocator.h>

void* CpuAllocator::allocate(size_t bytesCount) {
	void* p = NULL;
	if (cudaMallocHost((void**)&p, bytesCount) != cudaSuccess) {
		throw std::bad_alloc();
	}
	return p;
}

void CpuAllocator::deallocate(void* p) {
	if (cudaFreeHost(p) != cudaSuccess) {
		throw std::bad_alloc();
	}
}

void CpuAllocator::operator()(void* p) {
	if (cudaFreeHost(p) != cudaSuccess) {
		throw std::bad_alloc();
	}
}

void CpuAllocator::Copy(unsigned char* dst, unsigned char* src, const size_t count) {
	memcpy(dst, src, count);
}