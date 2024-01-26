
#ifndef __GRAYSCALE_CORRECTION_OPTIONS__
#define __GRAYSCALE_CORRECTION_OPTIONS__

class GrayscaleCorrectionOptions {
private:
	unsigned MadIndex;

public:
	char *MatrixA;
	char *MatrixB;

	GrayscaleCorrectionOptions(void) {
		MadIndex = 0;
	};
	GrayscaleCorrectionOptions(const unsigned madIndex) {
		MadIndex = madIndex;
	};

	~GrayscaleCorrectionOptions(void) { };

	bool Parse(int argc, char *argv[]);
};

#endif // __GRAYSCALE_CORRECTION_OPTIONS__
