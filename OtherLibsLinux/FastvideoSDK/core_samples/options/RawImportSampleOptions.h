
#ifndef __RAW_IMPORT_SAMPLE_OPTIONS__
#define __RAW_IMPORT_SAMPLE_OPTIONS__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "RawImportOptions.h"

class RawImportSampleOptions : public virtual BaseOptions {
public:
	RawImportSampleOptions(void) {};
	~RawImportSampleOptions(void) {};

	RawImportOptions Raw;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __RAW_IMPORT_SAMPLE_OPTIONS__
