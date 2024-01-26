#include <cstdio>
#include <string>

#include <sstream>

#include "helper_raw.hpp"
#include "helper_dir.hpp"
#include "supported_files.hpp"

#include "FastAllocator.h"
#include "checks.h"

std::string generateOutputFileName(const char *pattern, unsigned idx) {
	std::string outputFileName(pattern);
	std::string::size_type loc = outputFileName.find("*", 0);
	if (loc != std::string::npos) {
		std::ostringstream convert;
		convert << idx;
		outputFileName.replace(loc, 1, convert.str());
	}
	return outputFileName;
}

template<typename T>
void CreateListForThread(std::list<T> &src, std::list<T> &dst, int index) {
	for (auto i = src.begin(); i != src.end(); i++) {
		T img = (*i);

		size_t found = img.outputFileName.rfind(".");
		if (found != std::string::npos) {
			img.outputFileName.replace(found, 1, "_" + std::to_string(index) + ".");
		}

		dst.push_back(img);
	}
}

bool IsGrayUncompressedFormat(const char* fname) {
	if (compareFileExtension(fname, ".pgm") || compareFileExtension(fname, ".tif") || compareFileExtension(fname, ".tiff") || compareFileExtension(fname, ".bmp"))
		return true;
	return false;
}

bool IsColorUncompressedFormat(const char* fname) {
	if (compareFileExtension(fname, ".ppm") || compareFileExtension(fname, ".tif") || compareFileExtension(fname, ".tiff") || compareFileExtension(fname, ".bmp"))
		return true;

	return false;
}

template void CreateListForThread(std::list<Bytestream<FastAllocator> > &src, std::list<Bytestream<FastAllocator>> &dst, int index);
template void CreateListForThread(std::list<Image<FastAllocator> >& src, std::list<Image<FastAllocator>>& dst, int index);
