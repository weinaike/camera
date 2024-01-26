#ifndef __MULTI_THREAD_INFO__
#define __MULTI_THREAD_INFO__

struct MtResult {
	double requestedMemSize;
	double totalTime;
	double totalFileSize;
	unsigned processedItem;

	double pipelineHostTime;
	double readerWaitTime;
	double writerWaitTime;
	double allocationTime;

	double componentTime;

	fastStatus_t lastStatus;
	volatile bool isCreated;
};

#endif // __MULTI_THREAD_INFO__
