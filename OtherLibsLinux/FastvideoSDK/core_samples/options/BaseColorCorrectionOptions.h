
#ifndef __BASE_COLOR_CORRECTION_OPTIONS__
#define __BASE_COLOR_CORRECTION_OPTIONS__

class BaseColorCorrectionOptions {
private:
	unsigned OptionIndex;

public:
	float BaseColorCorrection[12];
	bool BaseColorCorrectionEnabled;

	int WhiteLevel[3];
	bool WhiteLevelEnabled;

	BaseColorCorrectionOptions(void) {
		OptionIndex = 0;
	};
	BaseColorCorrectionOptions(const unsigned optionIndex) {
		OptionIndex = optionIndex;
	};
	~BaseColorCorrectionOptions(void) {};

	bool Parse(int argc, char *argv[]);
};

#endif // __BASE_COLOR_CORRECTION_OPTIONS__
