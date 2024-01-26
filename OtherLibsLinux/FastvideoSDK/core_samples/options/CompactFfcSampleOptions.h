
#pragma once

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "CompactFfcOptions.h"

class CompactFfcSampleOptions : public virtual BaseOptions {
public:
	CompactFfcSampleOptions(void) {};
	~CompactFfcSampleOptions(void) {};

	CompactFfcOptions CompactFfc;

	virtual bool Parse(int argc, char *argv[]);
};
