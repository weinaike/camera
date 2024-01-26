
#ifndef __NPP_IMAGE_FILTER_OPTIONS__
#define __NPP_IMAGE_FILTER_OPTIONS__

class NppImageFilterOptions {
public:
	static double DisabledConst;

	unsigned RawWidth;
	unsigned RawHeight;
	
	unsigned BitsCount;

	double Sigma;
	double Radius;
	double Amount;

	double envelopMedian;
	double envelopSigma;
	int envelopRank;
	double envelopCoof;

	double Threshold;

	NppImageFilterOptions(void) { };
	~NppImageFilterOptions(void) { };

	bool Parse(int argc, char *argv[]);
};

#endif // __NPP_IMAGE_FILTER_OPTIONS__
