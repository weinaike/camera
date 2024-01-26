
#pragma once

#include "fastvideo_decoder_j2k.h"

#include "J2kPrintBoxes.h"

void FreeJ2kImageInfo(fastJ2kImageInfo_t* info);
int WriteJ2kGML(const char* filename, fastJ2kImageInfo_t* info);
