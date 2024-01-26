#ifndef __HELPER_LUT__
#define __HELPER_LUT__

#include "fastvideo_sdk_define.h"
#include <memory>

template<class T, class Allocator> fastStatus_t
fvLoadLut(const char *file, std::unique_ptr<T, Allocator> &data, unsigned size);

#endif // __HELPER_LUT__
