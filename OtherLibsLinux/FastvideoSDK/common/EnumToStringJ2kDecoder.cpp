
#include "EnumToStringJ2kDecoder.h"

// Table A.10 in T-REC-T.800-201410 Amendment 7 - profiles for IMF (Interoperable master format)
const char* EnumToString(fastJ2kCapability_t value) {
	switch (value) {
		case FAST_J2K_CAPABILITY_ANY: // Any capabilities specified in this Recommendation | International Standard
			return "Any";
		case FAST_J2K_CAPABILITY_CSTREAM_RESTRICTION_0: // codestream restricted as described for Profile 0 from Table A.45
			return "Profile 0 restriction";
		case FAST_J2K_CAPABILITY_CSTREAM_RESTRICTION_1: // codestream restricted as described for Profile 1 from Table A.45
			return "Profile 1 restriction";
		case FAST_J2K_CAPABILITY_DCINEMA_2K: // specified in Table A.46
			return "2K Digital Cinema Profile";
		case FAST_J2K_CAPABILITY_DCINEMA_4K: // specified in Table A.46
			return "4K Digital Cinema Profile";
		case FAST_J2K_CAPABILITY_SCALABLE_DCINEMA_2K: // specified in Table A.46
			return "Scalable 2K Digital Cinema Profile";
		case FAST_J2K_CAPABILITY_SCALABLE_DCINEMA_4K: // specified in Table A.46
			return "Scalable 4K Digital Cinema ProfileK";
		case FAST_J2K_CAPABILITY_OTHER:
			return "Other";
		default:
			return "Other";
	}
}

const char* EnumToString(fastJ2kWavelet_t wt)
{
	switch (wt)
	{
		case FAST_J2K_WT_CDF97: return "CDF 9/7 (Lossy/Irreversible)";
		case FAST_J2K_WT_CDF53: return "CDF 5/3 (Lossless/Reversible)";
		case FAST_J2K_WT_CUSTOM: return "Custom";
		default: return "Unknown";
	}
}

const char* EnumToString(fastJ2kMCT_t mct) {

	switch (mct)
	{
		case FAST_J2K_MCT_NONE: return "None";
		case FAST_J2K_MCT_REVERSIBLE: return "Reversible";
		case FAST_J2K_MCT_IRREVERSIBLE: return "Irreversible";
		default: return "Unknown";
	}
}

const char* EnumToString(fastJ2kProgression_t prg) {
	switch (prg)
	{
		case FAST_J2K_PT_LRCP: return  "LRCP";
		case FAST_J2K_PT_RLCP: return "RLCP";
		case FAST_J2K_PT_RPCL: return "RPCL";
		case FAST_J2K_PT_PCRL: return "PCRL";
		case FAST_J2K_PT_CPRL: return "CPRL";
		default: return "Unknown";
	}
}
