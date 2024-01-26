

#ifndef __HISTOGRAM_OPTIONS_H__
#define __HISTOGRAM_OPTIONS_H__
#include "fastvideo_sdk_define.h"

class HistogramOptions {
public:
	int BinCount;
	int RoiLeftTopX;
	int RoiLeftTopY;
	int RoiWidth;
	int RoiHeight;

	int ColumnStride;

	fastHistogramType_t HistogramType;
	fastBayerPattern_t BayerPattern;

	HistogramOptions(void) { };
	~HistogramOptions(void) { };

	bool Parse(int argc, char *argv[]);

private:
	static fastBayerPattern_t GetBayerPatternFromString(const char *pattern);
	static fastHistogramType_t GetHistogramTypeFromString(const char *pattern);
};

#endif /* __HISTOGRAM_OPTIONS_H__ */
