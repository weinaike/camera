
#ifndef __RAW_IMPORT_OPTIONS__
#define __RAW_IMPORT_OPTIONS__

#include "fastvideo_sdk_define.h"

class RawImportOptions {
private:
	bool FormatParser(const char *rawFormat);

public:
	fastRawFormat_t RawFormat;

	unsigned Width;
	unsigned Height;
	
	bool IsGpu;
	bool IsConvert12to16;
	
	RawImportOptions(void) {};
	~RawImportOptions(void) {};

	bool Parse(int argc, char *argv[]);
};

#endif // __RAW_IMPORT_OPTIONS__
