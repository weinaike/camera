

#pragma once

#include "fastvideo_decoder_j2k.h"

int PrintJ2kGML(fastJ2kImageInfo_t* info);
void PrintJ2kReaderRequirementBox(fastJ2kImageInfo_t* info);
void PrintJ2kAsocBoxes(fastJ2kImageInfo_t* info);
void PrintJ2kUUIDboxes(fastJ2kImageInfo_t* info);
