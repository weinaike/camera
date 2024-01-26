#pragma once
#include "BaseOptions.h"
#include "AsyncReaderTaskEnqueuer.hpp"

inline AsyncReaderTaskEnqueuer* AsyncReaderTaskEnqueuerFactory(BaseOptions* options, IAsyncFileReader* fileReader, volatile bool* terminateAll) {
	AsyncReaderTaskEnqueuer* fileEnqueuer = nullptr;

	if (options->IsFolder) {
		AsyncReaderTaskEnqueuerForDirWithRepeat* dirFileEnqueuer = new AsyncReaderTaskEnqueuerForDirWithRepeat();
		dirFileEnqueuer->Init(options->InputPath, options->OutputPath, options->RepeatCount, fileReader, terminateAll);
		fileEnqueuer = dirFileEnqueuer;
	} else {
		AsyncReaderTaskEnqueuerForSingleWithRepeat* singleFileEnqueuer = new AsyncReaderTaskEnqueuerForSingleWithRepeat();
		singleFileEnqueuer->Init(options->InputPath, options->OutputPath, options->RepeatCount, fileReader, terminateAll);
		fileEnqueuer = singleFileEnqueuer;
	}
	return fileEnqueuer;
}
