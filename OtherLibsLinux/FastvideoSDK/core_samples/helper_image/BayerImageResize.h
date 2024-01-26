
#ifndef __BAYER_IMAGE_RESIZE__
#define __BAYER_IMAGE_RESIZE__

#include "FastAllocator.h"
#include "Image.h"

int BayerMergeLines(Image<FastAllocator> &img);
int BayerSplitLines(Image<FastAllocator> &img);

#endif // __BAYER_IMAGE_RESIZE__
