#include "EnumToStringJ2kEncoder.h"

const char* EnumToString(fastEncoderJ2kAlgorithmType_t value) {
	switch (value) {
		case FAST_J2K_ENCODER_ALGORITHM_UNKNOWN:
			return "Unknown";
		case FAST_J2K_ENCODER_ALGORITHM_REVERSIBLE:
			return "Reversible";
		case FAST_J2K_ENCODER_ALGORITHM_IRREVERSIBLE:
			return "Irreversible";
		default:
			return "Other";
	}
}
