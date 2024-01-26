#include "BatchedQueue.h"

template<typename T>
int BatchedQueue<T>::GetNextWriterIndex(int index) const {
	if (index + Writers >= size)
		return index + Writers - size;

	return index + Writers;
}

template<typename T>
int BatchedQueue<T>::GetNextReaderIndex(int index) const {
	if (index + Readers >= size)
		return index + Readers - size;

	return index + Readers;
}

template<typename T>
BatchedQueue<T>::BatchedQueue(
	int queueSize, int batchSize,
	volatile bool* terminate,
	int writers, int readers
) {
	size = queueSize;
	this->batchSize = batchSize;
	values = new Batch<T> * [queueSize];
	for (int i = 0; i < queueSize; i++) {
		values[i] = new Batch<T>(batchSize);
	}

	Terminate = terminate;
	Writers = writers;
	Readers = readers;

	writerIndexes = new int[Writers];
	for (int i = 0; i < Writers; i++) {
		writerIndexes[i] = i - Writers;
	}

	readerIndexes = new int[Readers];
	for (int i = 0; i < Readers; i++) {
		readerIndexes[i] = i - Readers;
	}

	writerFinished = new bool[Writers];
	for (int i = 0; i < Writers; i++) {
		writerFinished[i] = false;
	}
}

template<typename T>
void BatchedQueue<T>::Reinit() {
	for (int i = 0; i < Writers; i++) {
		writerIndexes[i] = i - Writers;
	}
	for (int i = 0; i < Readers; i++) {
		readerIndexes[i] = i - Readers;
	}

	for (int i = 0; i < size; i++) {
		values[i]->isFree = true;
	}
}

template<typename T>
Batch<T>* BatchedQueue<T>::At(int i) {
	if (i < size)
		return values[i];
	return nullptr;
}

template<typename T>
BatchedQueue<T>::~BatchedQueue() {
	for (int i = 0; i < size; i++) {
		delete values[i];
	}
	delete[] values;

	delete[] writerIndexes;
	delete[] readerIndexes;
	delete[] writerFinished;
}

template<typename T>
int BatchedQueue<T>::GetSize() const {
	return size;
}

template<typename T>
int BatchedQueue<T>::GetBatchSize() const {
	return batchSize;
}

template<typename T>
Batch<T>* BatchedQueue<T>::GetNextWriterBatch(int writerId) {
	int writerIndex = writerIndexes[writerId];
	int nextWriterIndex = GetNextWriterIndex(writerIndex);

	while (!values[nextWriterIndex]->isFree) {
		if (*Terminate)
			return nullptr;

		std::unique_lock<std::mutex> locker(values[nextWriterIndex]->lockBatch);
		if (values[nextWriterIndex]->isFree)
			break;
		values[nextWriterIndex]->becomeFree.wait(locker);
	}
	writerIndexes[writerId] = nextWriterIndex;

	return values[nextWriterIndex];
}

template<typename T>
void BatchedQueue<T>::CommitWriterBatch(int writerId) {
	int writerIndex = writerIndexes[writerId];
	std::unique_lock<std::mutex> locker(values[writerIndex]->lockBatch);

	values[writerIndex]->isFree = false;
	values[writerIndex]->becomeReady.notify_one();
}

template<typename T>
bool BatchedQueue<T>::CheckReaderBatchReady(int readerId, bool waitRequired) const {
	int readerIndex = readerIndexes[readerId];

	while (values[readerIndex]->isFree && waitRequired) {
		if (*Terminate)
			return false;
		std::unique_lock<std::mutex> locker(values[readerIndex]->lockBatch);
		values[readerIndex]->becomeReady.wait(locker);
	}

	return !values[readerIndex]->isFree;
}

template<typename T>
Batch<T>* BatchedQueue<T>::GetNextReaderBatch(int readerId, bool waitRequired) {
	int readerIndex = readerIndexes[readerId];
	int nextReaderIndex = GetNextReaderIndex(readerIndex);

	while (values[nextReaderIndex]->isFree && waitRequired && !AllWriterFinished()) {
		if (*Terminate)
			return nullptr;

		std::unique_lock<std::mutex> locker(values[nextReaderIndex]->lockBatch);
		if (!values[nextReaderIndex]->isFree)
			break;
		values[nextReaderIndex]->becomeReady.wait(locker);
	}
	if (AllWriterFinished() && values[nextReaderIndex]->isFree)
		return nullptr;

	readerIndexes[readerId] = nextReaderIndex;
	return values[nextReaderIndex];
}

template<typename T>
void BatchedQueue<T>::FreeReaderBatch(int readerId) {
	int readerIndex = readerIndexes[readerId];

	std::unique_lock<std::mutex> locker(values[readerIndex]->lockBatch);
	values[readerIndex]->isFree = true;
	values[readerIndex]->becomeFree.notify_one();
}

template<typename T>
void BatchedQueue<T>::WaitFirstQueueFill() {
	int readerIndex = readerIndexes[0];
	if (readerIndex < 0) {
		Batch<T>* batch = values[size - 2];
		while (batch->isFree) {
			if (*Terminate)
				return;
			std::unique_lock<std::mutex> locker(batch->lockBatch);
			batch->becomeReady.wait(locker);
		}

		batch = values[size - 1];
		while (batch->isFree) {
			if (*Terminate)
				return;
			std::unique_lock<std::mutex> locker(batch->lockBatch);
			batch->becomeReady.wait(locker);
		}
	}
}

template<typename T>
int BatchedQueue<T>::GetMaxItemCount() const {
	return batchSize * size;
}

template<typename T>
int BatchedQueue<T>::GetFilledItemCount() {
	int res = 0;
	for (int i = 0; i < size; i++) {
		if (!values[i]->isFree) {
			Batch<T>* batch = values[i];
			res += batch->GetSize();
		}
	}
	return res;
}

template<typename T>
int BatchedQueue<T>::GetWriterBatchFilledItemCount(int writerId) const {
	const int writerIndex = writerIndexes[writerId];
	
	int res = 0;
	Batch<T>* batch = values[writerIndex];
	for (int j = 0; j < batchSize; j++) {
		res += batch->GetSize();
	}
	return res;
}

template<typename T>
void BatchedQueue<T>::ReleaseAll() {
	for (int i = 0; i < size; i++) {
		Batch<T>* batch = values[i];
		std::unique_lock<std::mutex> locker(batch->lockBatch);
		batch->becomeReady.notify_one();
	}
	for (int i = 0; i < size; i++) {
		Batch<T>* batch = values[i];
		std::unique_lock<std::mutex> locker(batch->lockBatch);
		batch->becomeFree.notify_one();
	}
}

template<typename T>
void BatchedQueue<T>::WaitAllFree(int readerId) {
	int readerIndex = readerIndexes[readerId];
	int lastNonFreeIndex = -1;
	for (int i = 0; i < size; i++) {
		if (!values[readerIndex]->isFree) {
			lastNonFreeIndex = readerIndex;
		}
		readerIndex = GetNextReaderIndex(readerIndex);
	}

	if (lastNonFreeIndex > -1) {
		while (!values[lastNonFreeIndex]->isFree) {
			if (*Terminate)
				return;
			std::unique_lock<std::mutex> locker(values[lastNonFreeIndex]->lockBatch);
			values[lastNonFreeIndex]->becomeFree.wait(locker);
		}
	}
}

template<typename T>
void BatchedQueue<T>::WriterFinished(int writerId)
{
	writerFinished[writerId] = true;
	if (AllWriterFinished())
	{
		for (int readerId = 0; readerId < Readers; readerId++)
		{
			int readerIndex = readerIndexes[readerId];
			int nextReaderIndex = GetNextReaderIndex(readerIndex);
			values[nextReaderIndex]->becomeReady.notify_one();
		}
	}
}

template<typename T>
bool BatchedQueue<T>::AllWriterFinished()
{
	bool r = true;
	for (int i = 0; i < Writers; i++)
		r = r && writerFinished[i];
	return r;
}
