#ifndef __HELPER_PFM__
#define __HELPER_PFM__

#include "fastvideo_sdk_define.h"

template<class Allocator> fastStatus_t
fvLoadPFM(const char *file, std::unique_ptr<float, Allocator> &data,
unsigned int &width, unsigned int &pitch, unsigned int pitchAlignment,
		unsigned int &height, unsigned &channels
);

fastStatus_t fvSavePFM(const char *file, float *data, unsigned int width, unsigned int pitch, unsigned int height, unsigned int channels);

#endif // __HELPER_PFM__
