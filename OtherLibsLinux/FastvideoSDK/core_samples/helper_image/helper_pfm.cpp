#include "helper_ppm.hpp"
#include "FastAllocator.h"
#include "checks.h"
#include "stdio.h"

fastStatus_t fvSavePFM(const char *file, float *data, unsigned int width, unsigned int pitch, unsigned int height, unsigned int channels) {
	FILE *fp = fopen(file, "wb+");
	if (fp == NULL) {
		return FAST_IO_ERROR;
	}

	// check header
	char header[3] = "P7";
	if (fwrite(header, sizeof(char), 2, fp) < 2) {
		return FAST_IO_ERROR;
	}

	fwrite(&width, sizeof(unsigned int), 1, fp);
	fwrite(&height, sizeof(unsigned int), 1, fp);
	fwrite(&channels, sizeof(unsigned int), 1, fp);

	// read and close file
	for (unsigned y = 0; y < height; y++) {
		if (fwrite(&data[y * pitch], sizeof(float), width * channels, fp) == 0) {
			return FAST_IO_ERROR;
		}
	}

	fclose(fp);
	return FAST_OK;
}

template<class Allocator>
fastStatus_t fvLoadPFM(const char *file, std::unique_ptr<float, Allocator> &data,
	unsigned int &width, unsigned int &pitch, unsigned int pitchAlignment,
	unsigned int &height, unsigned &channels
) {
	FILE *fp = fopen(file, "rb");
	if (fp == NULL) {
		return FAST_IO_ERROR;
	}

	// check header
	char header[2] = { 0 };
	if (fread(header, sizeof(char), 2, fp) < 2 ||
		(header[0] != 'P' && header[1] != '7')) {
		return FAST_IO_ERROR;
	}

	unsigned int fileParameters[3];
	if (fread(fileParameters, sizeof(unsigned int), 3, fp) < 3) {
		return FAST_IO_ERROR;
	}

	width = fileParameters[0];
	height = fileParameters[1];
	channels = fileParameters[2];
	pitch = width * channels;

	Allocator alloc;
	CHECK_FAST_ALLOCATION(data.reset((float *)alloc.allocate(height * pitch * sizeof(float))));

	// read and close file
	for (unsigned y = 0; y < height; y++) {
		if (fread(&data.get()[y * pitch], sizeof(float), width * channels, fp) == 0) {
			return FAST_IO_ERROR;
		}
	}

	fclose(fp);
	return FAST_OK;
}

template fastStatus_t fvLoadPFM(const char *file, std::unique_ptr<float, FastAllocator> &data,
	unsigned int &width, unsigned int &pitch, unsigned int pitchAlignment,
	unsigned int &height, unsigned &channels
);
