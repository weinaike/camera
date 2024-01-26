#ifndef __GPU_INFO__
#define __GPU_INFO__

#include "fastvideo_sdk_define.h"

fastStatus_t setGpuDeviceId(int deviceId);

fastStatus_t GetNvmlMemoryInfo(int cudaDeviceId, size_t* freeMemory, size_t* totalMemory);

#endif // __GPU_INFO__
