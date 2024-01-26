
#pragma once
#include "fastvideo_sdk_define.h"
#include "supported_files.hpp"
#include "helper_bytestream.hpp"
#include "helper_jpeg.hpp"

fastStatus_t PreloadJpegFromFolder(const char* folderName, fastJfifInfo_t* jfifInfo);

fastStatus_t PreloadJpegFromFile(const char* fileName, fastJfifInfo_t* jfifInfo);

fastStatus_t PreloadJpeg(const char* path, bool isFolder, fastJfifInfo_t* jfifInfo);
