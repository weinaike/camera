#pragma once

#include "Image.h"

class IAsyncFileReader {
public:
	virtual ~IAsyncFileReader() {}
	virtual void EnqueNextFile(File_t* fileInfo) = 0;
	virtual void WaitAll() = 0;
};
