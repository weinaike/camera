#ifndef __FilesParameterChecker_HPP__
#define __FilesParameterChecker_HPP__

class FilesParameterChecker {
public:
	typedef enum {
		FAST_RAW,
		FAST_RAW_COLOR,
		FAST_RAW_GRAY,
		FAST_BMP,
		FAST_JPEG,
		FAST_JPEG2000,
		FAST_AVI,
		FAST_MP4,
		FAST_MXF,
		FAST_YV12,
		FAST_SDI,
		FAST_GRAY_COLOR,
	} fastFileType_t;

	typedef enum {
		FAST_OK,
		FAST_INPUT_ERROR,
		FAST_OUTPUT_ERROR,
		FAST_BOTH_ERROR
	} fastFileStatus_t;

private:
	static bool IsRaw(const char *fileName);
	static bool IsBmp(const char *fileName);
	static bool IsRawColor(const char *fileName);
	static bool IsRawGray(const char *fileName);
	static bool IsJpeg(const char *fileName);
	static bool IsAvi(const char *fileName);
	static bool IsMxf(const char *fileName);
	static bool IsMp4(const char *fileName);
	static bool IsJPEG2000(const char *fileName);
	static bool IsYV12(const char *fileName);
	static bool IsSDI(const char *fileName);
	static bool IsGrayColor(const char *fileName);
	static bool IsValid(const char *fileName, const fastFileType_t fileType);

public:
	static bool CheckFileExtension(const char *fileName, const char *etalon);
	static fastFileStatus_t Validate(const char *inputFile, fastFileType_t inputFileType, const char *outputFile, fastFileType_t outputFileType);
	static fastFileStatus_t Validate(const char * inputFile, fastFileType_t  inputFileType);
};

#endif	// __FilesParameterChecker_HPP__
