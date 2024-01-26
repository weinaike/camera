
#include "NppImageFilterOptions.h"
#include "ParametersParser.h"

#include <cstdio>

double NppImageFilterOptions::DisabledConst = -1.0;

bool NppImageFilterOptions::Parse(int argc, char *argv[]) {
	RawWidth = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "w");
	RawHeight = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "h");
	BitsCount = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "bits");
	if (BitsCount != 8 && BitsCount != 12) {
		BitsCount = 8;
	}

	Sigma = DisabledConst;
	if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "sigma")) {
		Sigma = ParametersParser::GetCmdLineArgumentFloat(argc, const_cast<const char **>(argv), "sigma", DisabledConst);
		if (Sigma < 0.) {
			fprintf(stderr, "Incorrect sigma = %.3f. Set to 1\n", Sigma);
			Sigma = 1.;
		}
	}

	Radius = ParametersParser::GetCmdLineArgumentFloat(argc, const_cast<const char **>(argv), "radius", 1.);
	if (Radius < 0.) {
		fprintf(stderr, "Incorrect radius = %.3f. Set to 1\n", Radius);
		Radius = 1.;
	}

	Amount = DisabledConst;
	if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "amount")) {
		Amount = ParametersParser::GetCmdLineArgumentFloat(argc, const_cast<const char **>(argv), "amount", DisabledConst);
		if (Amount < 0.) {
			Amount = 1.;
		}
	}

	envelopMedian = 0.5;
	if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "envelopMedian")) {
		envelopMedian = ParametersParser::GetCmdLineArgumentFloat(argc, const_cast<const char **>(argv), "envelopMedian", 0.5);
		if (envelopMedian <= 0. || envelopMedian >= 1.0) {
			fprintf(stderr, "Incorrect envelopMedian = %.3f. Should be (0;1). Set to 0.5\n", envelopMedian);
			envelopMedian = 0.5;
		}
	}

	envelopSigma = 0.5;
	if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "envelopSigma")) {
		envelopSigma = ParametersParser::GetCmdLineArgumentFloat(argc, const_cast<const char **>(argv), "envelopSigma", 0.5);
		if (envelopSigma <= 0.) {
			fprintf(stderr, "Incorrect envelopSigma = %.3f. Should be (0;). Set to 0.5\n", envelopSigma);
			envelopSigma = 0.5;
		}
	}

	envelopRank = 2;
	if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "envelopRank")) {
		envelopRank = ParametersParser::GetCmdLineArgumentInt(argc, const_cast<const char **>(argv), "envelopRank");
		if (envelopRank < 2 || envelopRank%2 == 1) {
			fprintf(stderr, "Incorrect envelopRank = %.3f. Should be (2;) and even. Set to 2\n", static_cast<double>(envelopRank));
			envelopRank = 2;
		}
	}

	envelopCoof = -0.01;
	if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "envelopCoof")) {
		envelopCoof = ParametersParser::GetCmdLineArgumentFloat(argc, const_cast<const char **>(argv), "envelopCoof", -0.01);
		if (envelopCoof >= 0.) {
			fprintf(stderr, "Incorrect envelopCoof = %.3f. Should be less 0. Set to -0.01\n", envelopCoof);
			envelopCoof = -0.01;
		}
	}

	Threshold = 0.;
	if (ParametersParser::CheckCmdLineFlag(argc, const_cast<const char **>(argv), "threshold")) {
		Threshold = ParametersParser::GetCmdLineArgumentFloat(argc, const_cast<const char **>(argv), "threshold", 0.);
		if (Threshold <= 0. || Threshold >= 1.0) {
			fprintf(stderr, "Incorrect threshold = %.3f. Should be (0;1). Set to 0.\n", Threshold);
			Threshold = 0.;
		}
	}

	return true;
}
