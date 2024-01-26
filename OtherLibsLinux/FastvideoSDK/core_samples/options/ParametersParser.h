
#ifndef __PARAMETERS_PARSER__
#define __PARAMETERS_PARSER__

class ParametersParser
{
private:
	static int StringRemoveDelimiter(char delimiter, const char *string);

public:
	ParametersParser(void) { };
	~ParametersParser(void) { };

	static int GetFileExtension(char *filename, char **extension);
	static bool CheckCmdLineFlag(const int argc, const char **argv, const char *string_ref);
	static bool GetCmdLineArgumentString(const int argc, const char **argv, const char *string_ref, char **string_retval);
	static int GetCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref);
	static int GetCmdLineArgumentInt(const int argc, const char **argv, const char *string_ref, const int NotFoundValue);
	static double GetCmdLineArgumentFloat(const int argc, const char **argv, const char *string_ref, const double NotFoundValue = 0.0f);
};

#endif // __PARAMETERS_PARSER__
