#pragma once

#include <condition_variable>

#include "Batch.hpp"
#undef max
#include <algorithm>

static unsigned CheckAndUpdateQueueSize(unsigned initialQueueSize, unsigned readerCount, unsigned writerCount)
{
	unsigned maxValue = std::max(readerCount, writerCount);
	for (unsigned i = 0; i < maxValue; i++)
	{
		unsigned val = initialQueueSize + i;
		if (val % readerCount == 0 && val % writerCount == 0)
			return val;
	}
	return initialQueueSize;
}

template<typename T>
class BatchedQueue {
private:
	Batch<T>** values;
	int size;
	int batchSize;

	int* writerIndexes;
	int* readerIndexes;
	int Writers;
	int Readers;

	volatile bool* Terminate;
	volatile bool* writerFinished;

	int GetNextWriterIndex(int index) const;
	int GetNextReaderIndex(int index) const;

public:
	BatchedQueue(int queueSize, int batchSize, volatile bool* terminate, int writers, int readers);
	~BatchedQueue();

	void Reinit();

	Batch<T>* At(int i);
	void CommitWriterBatch(int writerId);

	int GetSize() const;
	int GetBatchSize() const;

	bool CheckReaderBatchReady(int readerId, bool waitRequired) const;
	int GetWriterBatchFilledItemCount(int writerId) const;

	Batch<T>* GetNextWriterBatch(int writerId);
	Batch<T>* GetNextReaderBatch(int readerId, bool waitRequired = true);

	int GetMaxItemCount() const;
	int GetFilledItemCount();

	void WaitFirstQueueFill();
	void WaitAllFree(int readerId);

	void FreeReaderBatch(int readerId);

	void ReleaseAll();

	void WriterFinished(int writerId);
	bool AllWriterFinished();
};
