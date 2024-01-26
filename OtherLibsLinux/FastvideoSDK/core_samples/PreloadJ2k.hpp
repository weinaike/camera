
#pragma once
#include "fastvideo_sdk_define.h"
#include "supported_files.hpp"
#include "helper_bytestream.hpp"

#include "fastvideo_decoder_j2k.h"

fastStatus_t PreloadJ2kFromFolder(const char* folderName, fastJ2kImageInfo_t *j2kInfo);

fastStatus_t PreloadJ2kFromFile(const char* fileName, fastJ2kImageInfo_t* j2kInfo);

fastStatus_t PreloadJ2k(const char* path, bool isFolder, fastJ2kImageInfo_t* j2kInfo);
