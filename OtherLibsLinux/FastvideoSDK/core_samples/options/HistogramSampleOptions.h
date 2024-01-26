

#ifndef __HISTOGRAM_SAMPLE_OPTIONS_H__
#define __HISTOGRAM_SAMPLE_OPTIONS_H__

#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"
#include "HistogramOptions.h"

class HistogramSampleOptions : public virtual BaseOptions {
public:
	HistogramSampleOptions(void) {};
	~HistogramSampleOptions(void) {};

	HistogramOptions Histogram;

	virtual bool Parse(int argc, char *argv[]);
};

#endif
