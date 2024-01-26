
#ifndef __HELPER_LUT_3D__
#define __HELPER_LUT_3D__

#include "fastvideo_sdk_define.h"
#include "SampleTypes.h"

fastStatus_t fvLoadCube3D(
	const char *file,
	std::unique_ptr<float, FastAllocator> &data_r,
	std::unique_ptr<float, FastAllocator> &data_g,
	std::unique_ptr<float, FastAllocator> &data_b,
	unsigned &size
);
fastStatus_t fvSaveCube3D(
	const char *file,
	std::unique_ptr<float, FastAllocator> &data_r,
	std::unique_ptr<float, FastAllocator> &data_g,
	std::unique_ptr<float, FastAllocator> &data_b,
	unsigned size,
	unsigned bitsPerChannel
);

fastStatus_t fvLoadXml3D(
	const char *file,
	std::unique_ptr<float, FastAllocator> &data_h,
	std::unique_ptr<float, FastAllocator> &data_s,
	std::unique_ptr<float, FastAllocator> &data_v,
	fast_uint3* size,
	fastColorSaturationOperationType_t operationType[3]
);
fastStatus_t fvSaveXml3D(
	const char *file,
	std::unique_ptr<float, FastAllocator> &data_h,
	std::unique_ptr<float, FastAllocator> &data_s,
	std::unique_ptr<float, FastAllocator> &data_v,
	fast_uint3 &size,
	fastColorSaturationOperationType_t operationType[3]
);

#endif // __HELPER_LUT__
