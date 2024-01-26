
#pragma once

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "HdrImportOptions.h"

class HdrImportSampleOptions : public virtual BaseOptions {
public:
	HdrImportSampleOptions(void) : BaseOptions(false, false, false) {
	}
	~HdrImportSampleOptions(void) {
	}

	HdrImportOptions HdrImport;

	virtual bool Parse(int argc, char *argv[]);
};
