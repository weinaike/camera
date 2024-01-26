#pragma once
#include "fast_enums.h"

size_t GetPitchFromSurfaceInternal(const fastSurfaceFormat_t surfaceFmt, const unsigned width);
unsigned GetBytesPerChannelFromSurfaceInternal(const fastSurfaceFormat_t surfaceFmt);
unsigned GetNumberOfChannelsFromSurfaceInternal(const fastSurfaceFormat_t surfaceFmt);
unsigned GetBitsPerChannelFromSurfaceInternal(const fastSurfaceFormat_t surfaceFmt);
size_t GetBufferSizeFromSurfaceInternal(const fastSurfaceFormat_t surfaceFmt, const unsigned width, const unsigned height);
