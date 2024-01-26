#include "fastvideo_sdk_define.h"

#include "BaseOptions.h"

#include "Help.h"
#include "GPU.hpp"
#include "DecodeError.hpp"

#include "LicenseInfoPrinter.hpp"
#include "checks.h"

bool sampleInit(int argc, char *argv[], BaseOptions &options) {
	if (!options.Parse(argc, argv)) {
		helpPrint();
		return false;
	}

	if (options.Help) {
		helpPrint();
		return false;
	}

	if (!LicenseInfoPrinter()) {
		return false;
	}

	if (!DecodeError(setGpuDeviceId(options.DeviceId))) {
		return false;
	}

	if (options.LogFile != nullptr) {
		CHECK_FAST(fastTraceCreate(options.LogFile));
		CHECK_FAST(fastTraceEnableLUTDump(true));
	}
	
	return true;
}

bool sampleDestroy(BaseOptions &options) {
	if (options.LogFile != nullptr) {
		CHECK_FAST(fastTraceClose());
	}

	return true;
}
