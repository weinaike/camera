#include "fastvideo_sdk_define.h"

#include <cstdio>

#include "SurfaceTraits.hpp"

template<typename T>
int DumpImageToText(
	fastSurfaceFormat_t surfaceFmt,

	T *img,
	const unsigned pitch,

	const unsigned width,
	const unsigned height,

	const char *fname
) {
	FILE *fp = fopen(fname, "w+");
	if (!fp) {
		fprintf(stderr, "Can not create output file\n");
		return -1;
	}

	for (unsigned i = 0; i < height; i++) {
		for (unsigned j = 0; j < width * GetNumberOfChannelsFromSurface(surfaceFmt); j++) {
			fprintf(fp, "%10d", img[i * pitch + j]);
		}
		fprintf(fp, "\n");
	}

	fclose(fp);
	return 0;
}

template<typename T>
int DumpImageChannelToText(
	T *img,
	const unsigned pitch,

	const unsigned width,
	const unsigned height,

	const char *fnameR,
	const char *fnameG,
	const char *fnameB
) {
	FILE *fpR = fopen(fnameR, "w+");
	FILE *fpG = fopen(fnameG, "w+");
	FILE *fpB = fopen(fnameB, "w+");
	if (!fpR || !fpG || !fpB) {
		fprintf(stderr, "Can not create output file\n");
		return -1;
	}

	for (unsigned i = 0; i < height; i++) {
		for (unsigned j = 0; j < width; j++) {
			fprintf(fpR, "%10d", img[i * pitch + j * 3 + 0]);
			fprintf(fpG, "%10d", img[i * pitch + j * 3 + 1]);
			fprintf(fpB, "%10d", img[i * pitch + j * 3 + 2]);
		}
		fprintf(fpR, "\n");
		fprintf(fpG, "\n");
		fprintf(fpB, "\n");
	}

	fclose(fpR);
	fclose(fpG);
	fclose(fpB);
	return 0;
}

template int DumpImageChannelToText<unsigned short>(
	unsigned short *img,
	const unsigned pitch,

	const unsigned width,
	const unsigned height,

	const char *fnameR,
	const char *fnameG,
	const char *fnameB
);
