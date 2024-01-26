#pragma once
#include "BaseOptions.h"
#include "MultiThreadInfo.hpp"
#include "barrier.hpp"

struct AsyncProcessorThreadParam {
	MtResult result;
	int threadId;
};

class IAsyncProcessor {
public:
	virtual ~IAsyncProcessor() {}
	virtual AsyncProcessorThreadParam* GetResults() = 0;
	virtual void WaitAll() = 0;
};

template<typename Pipeline, typename SourceType, typename DestinationType>
class AsyncProcessor: public  IAsyncProcessor  {
protected:
	unsigned int processorThreadCount;
	AsyncProcessorThreadParam* params;
	BaseOptions* options;
	barrier* pipelineCreatingSync;
	volatile bool* terminateAll;
	SourceType* srcReader;
	DestinationType* dstWriter;
	
	void* specialParams;
	std::thread** processorThreads;

	Pipeline** pipelines;

	fastStatus_t PipelineCall(
		AsyncProcessorThreadParam* data
	) {
		fastStatus_t ret = pipelines[data->threadId]->Init(options, &data->result, data->threadId, specialParams);
		if (ret != FAST_OK) {
			if (ret == FAST_INSUFFICIENT_DEVICE_MEMORY)
				fprintf(stderr, "Insufficient device memory in thread %d.\n", data->threadId);

			data->result.isCreated = false;
			return ret;
		}

		data->result.isCreated = true;
		pipelineCreatingSync->wait();

		CHECK_FAST(pipelines[data->threadId]->Transform(srcReader, dstWriter, data->threadId, &data->result, terminateAll, specialParams));
		

		return FAST_OK;
	}

	void* AsyncProcessorThreadFunc(AsyncProcessorThreadParam* data) {
		data->result.lastStatus = PipelineCall(data);
		return NULL;
	}

public:
	AsyncProcessor() {
		params = nullptr;
		options = nullptr;
		pipelineCreatingSync = nullptr;
		terminateAll = nullptr;
		srcReader = nullptr;
		dstWriter = nullptr;
		specialParams = nullptr;
	}

	fastStatus_t Init(
		unsigned int processorThreadCount,
		BaseOptions* options,
		SourceType* srcReader,
		DestinationType* dstWriter,
		volatile bool* terminateAll,
		void* specialParams = nullptr
	) {
		this->processorThreadCount = processorThreadCount;
		this->options = options;
		this->srcReader = srcReader;
		this->dstWriter = dstWriter;
		this->terminateAll = terminateAll;
		this->specialParams = specialParams;

		params = new AsyncProcessorThreadParam[processorThreadCount];
		pipelineCreatingSync = new barrier(processorThreadCount + 1);

		for (int i = 0; i < processorThreadCount; i++) {
			params[i].threadId = i;
			params[i].result.isCreated = false;
		}

		pipelines = new Pipeline* [processorThreadCount];
		processorThreads = new std::thread* [processorThreadCount];
		for (int i = 0; i < processorThreadCount; i++) {
			pipelines[i] = new Pipeline();
			processorThreads[i] = new std::thread(&AsyncProcessor::AsyncProcessorThreadFunc, this, &params[i]);
		}

		pipelineCreatingSync->wait();

		bool isAllCreatedSuccess = true;

		for (int i = 0; i < processorThreadCount; i++) {
			isAllCreatedSuccess = isAllCreatedSuccess && params[i].result.isCreated;
		}
		return isAllCreatedSuccess ? FAST_OK : FAST_INSUFFICIENT_DEVICE_MEMORY;
	}

	void WaitAll() {
		for (int i = 0; i < processorThreadCount; i++) {
			processorThreads[i]->join();
		}
	}
	AsyncProcessorThreadParam* GetResults() {
		return params;
	}

	~AsyncProcessor () {

		for (int i = 0; i < processorThreadCount; i++) {
			pipelines[i]->Close();
		}

		delete[] params;
		delete pipelineCreatingSync;
		for (int i = 0; i < processorThreadCount; i++) {
			delete pipelines[i];
			delete processorThreads[i];
		}
		delete[] pipelines;
		delete[] processorThreads;
	}
};
