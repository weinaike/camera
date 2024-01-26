
#ifndef __HELPER_EXIF__
#define __HELPER_EXIF__

#include <cstddef>

#include "libexif/exif-data.h"

#define EXIF_SECTION_CODE 0xFFE1

ExifEntry* fastExifInitTag(ExifData *exif, ExifIfd ifd, ExifTag tag);
ExifEntry* fastExifCreateTag(ExifData *exif, ExifIfd ifd, ExifTag tag, ExifFormat fmt, std::size_t len);
ExifEntry* fastExifCreateAsciiTag(ExifData *exif, ExifIfd ifd, ExifTag tag, unsigned int strlen);
ExifEntry* fastExifCreateUndefinedTag(ExifData *exif, ExifIfd ifd, ExifTag tag, unsigned int len);

#endif // __HELPER_EXIF__
