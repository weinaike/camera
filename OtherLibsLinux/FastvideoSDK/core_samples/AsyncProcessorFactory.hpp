#pragma once

#include "ManagedConstFastAllocator.hpp"
#include "ManagedFastAllocator.hpp"
#include "CollectionFastAllocator.hpp"

#include "AsyncReaderTaskEnqueuer.hpp"
#include "AsyncFileReader.hpp"
#include "AsyncFileWriter.hpp"
#include "AsyncProcessor.hpp"

#include "PreloadUncompressedImage.hpp"
#include "MultiThreadInfoPrinter.h"
#include "AsyncReaderTaskEnqueuerFactory.hpp"
#include "AsyncSingleFileReader.hpp"


template<
	typename ProcessorForDir, typename ProcessorForSingle, 
	typename DirReader, typename SingleReader, 
	typename Writer>
void AsyncProcessorFactory(
	AsyncReaderTaskEnqueuer** fileEnqueuer,
	IAsyncFileReader** reader, 
	IAsyncProcessor** processor,
	
	Writer *writer, 
	BaseOptions* options, 
	unsigned batchSize, 
	unsigned  queueSize,  
	unsigned fileReaderThreadCount,
	unsigned processorThreadCount, 
	volatile bool* terminateAll)
{

	if (options->IsFolder) {

		auto* imgReader = new DirReader();
		imgReader->Init(batchSize, queueSize, fileReaderThreadCount, processorThreadCount, terminateAll);

		AsyncReaderTaskEnqueuerForDirWithRepeat* dirFileEnqueuer = new AsyncReaderTaskEnqueuerForDirWithRepeat();
		dirFileEnqueuer->Init(options->InputPath, options->OutputPath, options->RepeatCount, imgReader, terminateAll);
		*fileEnqueuer = dirFileEnqueuer;
		auto dirprocessor = new AsyncProcessor<ProcessorForDir, DirReader, Writer>();

		fastStatus_t status = dirprocessor->Init(processorThreadCount, options, imgReader, writer, terminateAll);
		if (status != FAST_OK)
			*terminateAll = true;
		*reader = imgReader;
		*processor = dirprocessor;
	}
	else {

		auto* imgReader = new SingleReader();
		imgReader->Init(options->InputPath, options->OutputPath, options->RepeatCount, batchSize, processorThreadCount);

		auto singleprocessor = new AsyncProcessor<ProcessorForSingle, SingleReader,	Writer>();
		*fileEnqueuer = nullptr;

		fastStatus_t status = singleprocessor->Init(processorThreadCount, options, imgReader, writer, terminateAll);
		if (status != FAST_OK)
			*terminateAll = true;

		*reader = imgReader;
		*processor = singleprocessor;
	}
}
