#ifndef __BASE_OPTIONS__
#define __BASE_OPTIONS__

#include "fastvideo_sdk_define.h"

class BaseOptions {
	bool IsMustBeConverted(const char *fname);

public:
	char *InputPath;
	char *OutputPath;
	bool IsFolder;

	bool AsyncMode;

	unsigned RawWidth;
	unsigned RawHeight;

	unsigned BitsPerChannel;
	char *Lut;

	bool Info;
	bool BenchmarkInfo;
	bool Help;
	bool ConvertToBGR;
	bool ExclusiveMode;

	unsigned DeviceId;

	unsigned MaxWidth;
	unsigned MaxHeight;

	fastSurfaceFormat_t SurfaceFmt;

	unsigned RepeatCount;
	unsigned NumberOfThreads;

	unsigned NumberOfReaderThreads;
	unsigned NumberOfWriterThreads;

	bool MultiProcess;

	char *LogFile;

	bool IgnoreInput;
	bool IgnoreOutput;
	bool IgnoreMaxDimension;

	bool Discard;
	unsigned BatchSize;
	unsigned Queue;

	bool IsCpu;

	
	BaseOptions(void) {
		InputPath = nullptr;
		OutputPath = nullptr;
		IsFolder = false;

		RawWidth = RawHeight = 0;

		BitsPerChannel = 0;
		Lut = nullptr;

		Info = false;
		BenchmarkInfo = false;
		Help = false;
		ConvertToBGR = false;
		ExclusiveMode = false;

		DeviceId = 0;

		MaxWidth = MaxHeight = 0;

		SurfaceFmt = FAST_I8;

		RepeatCount = 0;
		NumberOfThreads = 1;
		NumberOfReaderThreads = 1;
		NumberOfWriterThreads = 1;

		MultiProcess = false;

		IgnoreInput = false;
		IgnoreOutput = false;
		IgnoreMaxDimension = false;
		ConvertToBGR = false;

		LogFile = nullptr;

		Discard = false;
		BatchSize = 1;
		Queue = 16;
		IsCpu = false;
	}

	BaseOptions(bool ignoreInput, bool ignoreOutput, bool ignoreMaxDimension) : BaseOptions() {
		IgnoreInput = ignoreInput;
		IgnoreOutput = ignoreOutput;
		IgnoreMaxDimension = ignoreMaxDimension;
	}

	~BaseOptions(void) {
	}

	virtual bool Parse(int argc, char *argv[]);

	static bool CheckFileExtension(const char *fileName, const char *etalon);
	static fastSurfaceFormat_t GetSurfaceFormatFromExtension(const char* fname);
};

#endif // __BASE_OPTIONS__
