
#include "StringToEnumDenoise.h"

#include <cstring>

fastDenoiseThresholdFunctionType_t IdentifyThresholdingFunction(char *functionName) {
	fastDenoiseThresholdFunctionType_t type = FAST_THRESHOLD_FUNCTION_UNKNOWN;

	if (strcmp(functionName, "HARD") == 0) type = FAST_THRESHOLD_FUNCTION_HARD;
	if (strcmp(functionName, "SOFT") == 0) type = FAST_THRESHOLD_FUNCTION_SOFT;
	if (strcmp(functionName, "GARROTE") == 0) type = FAST_THRESHOLD_FUNCTION_GARROTE;

	return type;
}

fastWaveletType_t IdentifyWavelet(char *EnumToString) {
	fastWaveletType_t type = FAST_WAVELET_CDF97;

	if (strcmp(EnumToString, "CDF97") == 0) type = FAST_WAVELET_CDF97;
	if (strcmp(EnumToString, "CDF53") == 0) type = FAST_WAVELET_CDF53;

	return type;
}
