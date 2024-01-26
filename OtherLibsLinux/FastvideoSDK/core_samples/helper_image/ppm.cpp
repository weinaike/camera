#include "ppm.h"
#include "helper_common.h"

#include <cmath>
#include <ctype.h>
#include <fstream>
#include <string.h>
#include <string>
#include <iostream>

#include "alignment.hpp"


int LoadHeader(
	std::ifstream& infile,
	unsigned& width,
	unsigned& height,
	unsigned& numberOfChannels,
	unsigned& bitsPerChannel,
	std::string& mMagic
) {
	infile >> mMagic;
	infile.seekg(1, infile.cur);

	// comments
	{
		char c;
		infile.get(c);
		if (c == '#') {
			while (c != '\n') {
				infile.get(c);
			}
		} else {
			infile.seekg(-1, infile.cur);
		}
	}

	int maxValue;
	infile >> width >> height >> maxValue;
	// skip delimiter
	infile.seekg(1, infile.cur);

	bitsPerChannel = int(log(maxValue) / log(2) + 1);

	if (mMagic == "P2" || mMagic == "P5" || mMagic == "P15") {
		numberOfChannels = 1;
	} else if (mMagic == "P6" || mMagic == "P16") {
		numberOfChannels = 3;
	} else {
		std::cout << "Not supported format" << std::endl;
		return -1;
	}

	return 0;
}

int LoadHeaderPPM(
	const char* fname,
	unsigned& width,
	unsigned& height,
	unsigned& numberOfChannels,
	unsigned& bitsPerChannel
) {
	std::ifstream infile(std::string(fname), std::ifstream::binary);
	if (!infile.is_open()) {
		std::cout << "Failed to open: " << fname << std::endl;
		return -1;
	}

	std::string mMagic;
	return LoadHeader(
		infile, width, height, numberOfChannels, bitsPerChannel, mMagic
	);
}

int LoadPPM(
	const char* file,
	void** data,
	BaseAllocator* alloc,
	unsigned int& width,
	unsigned& wPitch,
	unsigned int& height,
	unsigned& bitsPerChannel,
	unsigned& numberOfChannels
) {
	std::ifstream infile(std::string(file), std::ifstream::binary);
	if (!infile.is_open()) {
		std::cout << "Failed to open: " << file << std::endl;
		return -1;
	}

	std::string mMagic;
	if (LoadHeader(
		infile, width, height, numberOfChannels, bitsPerChannel, mMagic
	) < 0) {
		return -1;
	}

	const unsigned bytePerPixel = uDivUp<unsigned>(bitsPerChannel, 8);
	if (mMagic == "P2" || mMagic == "P5" || mMagic == "P15") {
		numberOfChannels = 1;
	} else if (mMagic == "P6" || mMagic == "P16") {
		numberOfChannels = 3;
	} else {
		std::cout << "Not supported format: " << file << std::endl;
		return -1;
	}

	wPitch = GetAlignedPitch(width, numberOfChannels, bytePerPixel, alloc->getAlignment());
	*data = alloc->allocate(wPitch * height);

	if (mMagic == "P2") {
		// ASCII
		unsigned short* d = static_cast<unsigned short*>(*data);
		for (unsigned i = 0; i < height; i++) {
			for (unsigned j = 0; j < width; j++) {
				std::string pixel_str;
				infile >> pixel_str;
				d[i * (wPitch >> 1) + j] = static_cast<unsigned short>(std::stoi(pixel_str));
			}
		}
	} else {
		// Binary
		bool fvHeader = mMagic == "P15" || mMagic == "P16";

		char* d = static_cast<char*>(*data);
		for (unsigned i = 0; i < height; i++) {
			infile.read(&d[i * wPitch], width * bytePerPixel * numberOfChannels);

			if (bytePerPixel == 2 && !fvHeader) {
				unsigned short* p = reinterpret_cast<unsigned short*>(&d[i * wPitch]);
				for (unsigned int x = 0; x < wPitch / bytePerPixel; x++) {
					unsigned short t = p[x];
					const unsigned short t1 = t >> 8;
					t = (t << 8) | t1;
					p[x] = t;
				}
			}
		}
	}

	return 1;
}

int SavePPM(
	const char *file,
	unsigned char *data,
	const unsigned w,
	const unsigned wPitch,
	const unsigned h,
	const int bitsPerChannel,
	const unsigned int channels
) {
	assert(NULL != data);
	assert(w > 0);
	assert(h > 0);

	std::fstream fh(file, std::fstream::out | std::fstream::binary);
	if (fh.bad())
		return 0;

	if (channels == 1) {
		fh << "P5\n";
	} else if (channels == 3) {
		fh << "P6\n";
	} else
		return 0;

	const unsigned long long size = ((unsigned long long)1 << bitsPerChannel) - 1;
	fh << w << "\n" << h << "\n" << size << std::endl;
	const int bytePerPixel = _uSnapUp<unsigned>(bitsPerChannel, 8) / 8;

	unsigned short * tmp = (unsigned short *)malloc(wPitch);
	for (unsigned int y = 0; y < h && fh.good(); y++) {
		if (bytePerPixel == 2) {
			unsigned short *p = (unsigned short*)data;
			for (unsigned int x = 0; x < wPitch / bytePerPixel; x++) {
				unsigned short t = p[y * wPitch / bytePerPixel + x];
				unsigned short t1 = t >> 8;
				t = (t << 8) | t1;
				tmp[x] = t;
			}
			fh.write(reinterpret_cast<const char *>(tmp), w * channels * bytePerPixel);
		} else
			fh.write(reinterpret_cast<const char *>(&data[y * wPitch]), w * channels * bytePerPixel);
	}

	free(tmp);

	fh.flush();
	if (fh.bad())
		return 0;

	fh.close();
	return 1;
}
