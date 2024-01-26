#pragma once
#include "fast_memory.h"
#include <cuda_runtime.h>

class FastAllocatorPinned {
public:
	typedef size_t		size_type;
	typedef const void*	const_pointer;

	template<class T>
	static T *allocate(size_t bytesCount, const_pointer = 0) {
		T* p = NULL;
		cudaError_t ret_cuda = cudaMallocHost( (void **)&p,  bytesCount * sizeof(T), cudaHostAllocMapped);
		if ( ret_cuda != cudaSuccess ) {
			throw std::bad_alloc();
		}
		return p;
	}

	static void deallocate(void* p, size_type) {
		cudaError_t ret_cuda = cudaFreeHost( p );
		if ( ret_cuda != cudaSuccess ) {
			throw std::bad_alloc();
		}
	}
	
	void operator()(void* p) {
		cudaError_t ret_cuda = cudaFreeHost( p );
		if ( ret_cuda != cudaSuccess ) {
			throw std::bad_alloc();
		}
	}
};
