
#ifndef __SDI_CONVERTER_OPTIONS__
#define __SDI_CONVERTER_OPTIONS__

#include "fastvideo_sdk_define.h"

class SDIConverterOptions {
private:
	bool FormatParser(const char *sdiFormat);

public:
	fastSDIFormat_t SDIFormat;
	fastRGBAAlphaPadding_t alphaPadding;
	bool IsGpu;

	unsigned Width;
	unsigned Height;
	
	bool IsExport;
	bool IsConvert12to16;

	char *FileNameAlternate;

	SDIConverterOptions(void) {};
	~SDIConverterOptions(void) {};

	bool Parse(int argc, char *argv[]);
};

#endif // __SDI_CONVERTER_OPTIONS__
