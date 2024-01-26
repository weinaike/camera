
#pragma once
#include "BaseAllocator.h"
#include "fastvideo_sdk_define.h"
#include "FastAllocator.h"

#include <vector>
#include <queue>

#include <cstdio>
#include <mutex>

template<typename Allocator>
class ConstMemoryManager {
	void** buffers;
	unsigned int count;
	size_t size;
	std::queue<void*> freeBuffers;
	bool isInited;
	std::mutex m;

public:
	ConstMemoryManager() {
		isInited = false;
		count = 0;
		size = 0;
		buffers = nullptr;
	}

	ConstMemoryManager(const ConstMemoryManager<Allocator> &copy) {
		isInited = copy.isInited;
		count = copy.count;
		size = copy.size;
		buffers = copy.buffers;
		for (int i = 0; i < count; i++) {
			buffers[i] = copy.buffers[i];
		}
		freeBuffers = copy.freeBuffers;
	}

	void init(unsigned int count, size_t bufferSizeinByte) {
		if (isInited && (this->count != count || this->size != bufferSizeinByte))
			freeAll();

		this->count = count;
		this->size = bufferSizeinByte;
		
		buffers = new void* [count];
		Allocator alloc;
		for (int i = 0; i < count; i++) {
			buffers[i] = alloc.allocate(bufferSizeinByte);
			freeBuffers.push(buffers[i]);
		}
		
		isInited = true;
	}

	bool IsInited() {
		return this->isInited;
	}

	void* allocate(size_t bytesCount) {
		std::lock_guard<std::mutex> lock(m);

		if (bytesCount > size)
			throw std::bad_alloc();

		if (!freeBuffers.size())
			throw std::bad_alloc();

		void* p = freeBuffers.front();
		freeBuffers.pop();
		return p;
	}

	void deallocate(void* p) {
		std::lock_guard<std::mutex> lock(m);
		freeBuffers.push(p);
	}

	void operator()(void* p) {
		deallocate(p);
	}

	void freeAll() {
		if (freeBuffers.size() != count)
			throw std::bad_alloc();

		Allocator alloc;
		for (int i = 0; i < count; i++) {
			alloc.deallocate(buffers[i]);
			freeBuffers.pop();
		}
		delete[] buffers;
	
		isInited = false;
	}

	~ConstMemoryManager() {
		//freeAll();
	}
};

extern std::vector<ConstMemoryManager<FastAllocator>> fastAllocatorConstManagers;

template<int id >
class ManagedConstFastAllocator : public BaseAllocator {
	
public:
	void initManager(unsigned int count, size_t bufferSizeinByte) {
		/*const int mgrIdx = id;
		printf("%d %d\n", mgrIdx, (int)fastAllocatorConstManagers.size() - 1 < mgrIdx);
		bool isResize = (int)fastAllocatorConstManagers.size() - 1 < mgrIdx;*/

		if ((int)fastAllocatorConstManagers.size() - 1 < id) {
			fastAllocatorConstManagers.resize(id + 1);
		}

		if (!fastAllocatorConstManagers[id].IsInited()) {
			fastAllocatorConstManagers[id].init(count, bufferSizeinByte);
		}
	}

	void* allocate(size_t bytesCount) {
		if (!fastAllocatorConstManagers[id].IsInited())
			throw std::bad_alloc();

		return fastAllocatorConstManagers[id].allocate(bytesCount);
	}

	void deallocate(void* p) {
		fastAllocatorConstManagers[id].deallocate(p);
	}

	void operator()(void* p) {
		fastAllocatorConstManagers[id].deallocate(p);
	}

	void freeAll() {
		fastAllocatorConstManagers[id].freeAll();
	}
};
