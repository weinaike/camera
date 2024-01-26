
#ifndef __CONVERTER_OPTIONS__
#define __CONVERTER_OPTIONS__

#include "fastvideo_sdk_define.h"

class ConverterOptions {
private:
	static fastRawFormat_t GetRawFormat(const char *str);

public:
	unsigned Shift;
	bool Randomize;

	fastRawFormat_t RawFormat;

	ConverterOptions(void) {};
	~ConverterOptions(void) {};

	bool Parse(int argc, char *argv[]);
};

#endif // __CONVERTER_OPTIONS__
