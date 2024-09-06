#pragma once
#include "cuda_runtime.h"

extern "C" void convert12BitGrayTo8BitRgb(const void* srcData, void * dstData, int width, int height);
extern "C" void convert8BitGrayToRgb(const void* srcData, void * dstData, int width, int height);