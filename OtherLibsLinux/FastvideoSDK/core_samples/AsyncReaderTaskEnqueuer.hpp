#pragma once

#include <thread>
#include "AsyncFileReader.hpp"
#include <list>
#include "supported_files.hpp"

class AsyncReaderTaskEnqueuer {
protected:
	volatile bool* terminateAll;
	IAsyncFileReader* fileReader;
	std::thread* enqueuerThread;

	unsigned repeat;

	virtual void EnqueuerFunc() = 0;
	
	void EnqueuerThreadFunc() {
		EnqueuerFunc();
	}

public:
	AsyncReaderTaskEnqueuer() { 
		enqueuerThread = nullptr;
		fileReader = nullptr;
		terminateAll = nullptr;
	}

	void Init(IAsyncFileReader *fileReader, unsigned repeat, volatile bool* terminateAll) {
		this->terminateAll = terminateAll;
		this->fileReader = fileReader;
		this->repeat = repeat;
		
		enqueuerThread = new std::thread(&AsyncReaderTaskEnqueuer::EnqueuerThreadFunc, this);
	}

	void WaitAll() {
		enqueuerThread->join();
	}

	~AsyncReaderTaskEnqueuer(){
		delete enqueuerThread;
	}
};

class AsyncReaderTaskEnqueuerForDirWithRepeat : public AsyncReaderTaskEnqueuer {
protected:
	const char* dirPath;
	const char* outPath;

	void EnqueuerFunc() {
		std::list<std::string> dirFiles;

		getFileList(dirPath, dirFiles);

		if (dirFiles.empty()) {
			fprintf(stderr, "No input files found\n");
			fileReader->EnqueNextFile(nullptr);
			return;
		}

		int idx = 0;
		for (auto file = dirFiles.begin(); file != dirFiles.end() && !(*terminateAll); file++)
		{
			for (int i = 0; i < repeat && !(*terminateAll); i++)
			{
				File_t fileTask;
				fileTask.inputFileName = *file;
				fileTask.outputFileName = generateOutputFileName(outPath, idx);
				idx++;
				fileReader->EnqueNextFile(&fileTask);
			}
		}
		fileReader->EnqueNextFile(nullptr);
	}
public:
	void Init(const char* dirPath, const char *outPath, unsigned repeat, IAsyncFileReader* fileReader, volatile bool* terminateAll)
	{
		AsyncReaderTaskEnqueuer::Init(fileReader, repeat, terminateAll);
		this->dirPath = dirPath;
		this->outPath = outPath;
		
	}
};

class AsyncReaderTaskEnqueuerForSingleWithRepeat : public AsyncReaderTaskEnqueuer
{
protected:
	const char* file;
	const char* outPath;

	void EnqueuerFunc()
	{
		int idx = 0;
		for (int i = 0; i < repeat && !(*terminateAll); i++)
		{
			File_t fileTask;
			fileTask.inputFileName = file;
			fileTask.outputFileName = generateOutputFileName(outPath, idx);
			idx++;
			fileReader->EnqueNextFile(&fileTask);
		}
		fileReader->EnqueNextFile(nullptr);
	}
public:
	void Init(const char* file, const char* outPath, unsigned repeat, IAsyncFileReader* fileReader, volatile bool* terminateAll)
	{
		AsyncReaderTaskEnqueuer::Init(fileReader, repeat, terminateAll);
		this->file = file;
		this->outPath = outPath;

	}

};
