#pragma once

#include "fastvideo_sdk_define.h"
#include "libavutil/pixfmt.h"

fastSurfaceFormat_t ConvertToSurfaceFmt(AVPixelFormat pixelFormat);
fastSDIFormat_t ConvertToSdiFmt(AVPixelFormat pixelFormat);
fastJpegFormat_t ConvertToSubsampling(AVPixelFormat pixelFormat);
unsigned ConvertToOutputBitDepth(AVPixelFormat pixelFormat);
