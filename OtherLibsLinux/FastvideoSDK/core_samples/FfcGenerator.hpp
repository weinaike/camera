#pragma once

#include "Image.h"
#include "FastAllocator.h"

#include "FfcSampleOptions.h"

unsigned CompactFFC(
    bool disableFiltering,
    Image<FastAllocator>& inputImg,
    Image<FastAllocator>& compactedFfc
);