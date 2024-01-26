

#include "HistogramOptions.h"

#include <cstdio>
#include <cstring>

#include "fastvideo_sdk_define.h"
#include "ParametersParser.h"

fastBayerPattern_t HistogramOptions::GetBayerPatternFromString(const char *pattern) {
	fastBayerPattern_t ret = FAST_BAYER_RGGB;

	if (pattern != NULL) {
		if (strcmp(pattern, "GRBG") == 0)
			ret = FAST_BAYER_GRBG;
		else if (strcmp(pattern, "BGGR") == 0)
			ret = FAST_BAYER_BGGR;
		else if (strcmp(pattern, "GBRG") == 0)
			ret = FAST_BAYER_GBRG;
		else if (strcmp(pattern, "RGGB") == 0)
			ret = FAST_BAYER_RGGB;
		else {
			fprintf(stderr, "Pattern %s was not recognized.\nSet to RGGB\n", pattern);
			ret = FAST_BAYER_RGGB;
		}
	}
	return ret;
}

fastHistogramType_t HistogramOptions::GetHistogramTypeFromString(const char *pattern) {
	fastHistogramType_t ret = FAST_HISTOGRAM_COMMON;

	if (pattern != NULL) {
		if (strcmp(pattern, "common") == 0)
			ret = FAST_HISTOGRAM_COMMON;
		else if (strcmp(pattern, "bayer") == 0)
			ret = FAST_HISTOGRAM_BAYER;
		else if (strcmp(pattern, "bayer_g1g2") == 0)
			ret = FAST_HISTOGRAM_BAYER_G1G2;
		else if (strcmp(pattern, "parade") == 0)
			ret = FAST_HISTOGRAM_PARADE;
		else {
			fprintf(stderr, "Pattern %s was not recognized.\nSet to COMMON\n", pattern);
			ret = FAST_HISTOGRAM_COMMON;
		}
	}
	return ret;
}

bool HistogramOptions::Parse(int argc, char *argv[]) {
	char *str = NULL;
	if (ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "bins", &str)) {
		BinCount = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "bins");
		if (BinCount == 0) {
			fprintf(stderr, "Bins number should be greater than zero.\n");
			return false;
		}
	} else {
		fprintf(stderr, "You should provide the bins count value.\n");
		return false;
	}

	RoiLeftTopX = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "sX");
	RoiLeftTopY = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "sY");
	RoiWidth = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "roiWidth");
	RoiHeight = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "roiHeight");

	/* Get type of histogram for calculation */
	ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "htype", &str);
	HistogramType = GetHistogramTypeFromString(str);

	/* Get Bayer pattern and number of channels for calculating histogram for Bayer images */
	if ((HistogramType == FAST_HISTOGRAM_BAYER) || (HistogramType == FAST_HISTOGRAM_BAYER_G1G2)) {
		ParametersParser::GetCmdLineArgumentString(argc, (const char **)argv, "pattern", &str);
		BayerPattern = GetBayerPatternFromString(str);
	} else {
		/* Ignore 'bayer' parameter for other histogram types and set default values */
		BayerPattern = FAST_BAYER_NONE;
	}
	/* Check the Bayer patter usage */
	if (BayerPattern != FAST_BAYER_NONE && HistogramType != FAST_HISTOGRAM_BAYER && HistogramType != FAST_HISTOGRAM_BAYER_G1G2) {
		fprintf(stderr, "Wrong context for Bayer pattern usage.\nBayer pattern parameter may only be used with appropriate histogram types.");
		return false;
	}

	/* Get column stride for Parade histogram calculation */
	ColumnStride = ParametersParser::GetCmdLineArgumentInt(argc, (const char **)argv, "cstride");
	if (ColumnStride < 1) {
		ColumnStride = 1;
	}

	return true;
}
