#pragma once

#include <condition_variable>

template<typename T> 
class Batch {
private:
	template<typename U> friend class BatchedQueue;
	unsigned int batchSize;
	T* values;
	bool isFree;

	unsigned int filledItem;
	std::mutex lockBatch;
	std::condition_variable becomeFree;
	std::condition_variable becomeReady;

public:
	Batch(int batchSize);
	~Batch();
	T* At(int i);
	unsigned int GetSize() const;

	unsigned int GetFilledItem() const;
	void SetFilltedItem(unsigned int count);
};
