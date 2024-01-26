

#include "helper_bytestream.hpp"

fastStatus_t fvSaveBytestream(
	std::string fname,
	unsigned char* inputImg,
	size_t size,
	bool info
) {
	hostTimer_t timer = NULL;
	if (info) {
		timer = hostTimerCreate();
		hostTimerStart(timer);
	}

	std::ofstream output(fname.c_str(), std::ofstream::binary);
	if (output.is_open()) {
		output.write((char*)inputImg, size);
		output.close();
	}
	else {
		return FAST_IO_ERROR;
	}

	if (info) {
		const double loadTime = hostTimerEnd(timer);
		printf("JFIF image write time = %.2f ms\n\n", loadTime * 1000.0);
		hostTimerDestroy(timer);
	}

	return FAST_OK;
}
