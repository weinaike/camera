
#ifndef __BMP_HPP__
#define __BMP_HPP__

#include "BaseAllocator.h"

////////////////////////////////////////////////////////////////////////////////
// BMP load / store
////////////////////////////////////////////////////////////////////////////////

typedef enum{
	BMP_OK,                           //No error
	BMP_IO_ERROR,                     //Failed to open/access file
	BMP_INVALID_FORMAT,               //Invalid file format
	BMP_UNSUPPORTED_FORMAT          //File format is unsupported by the current version of FAST
} bmpStatus_t;

int LoadHeaderBMP(
    const char* fname,
    unsigned& width,
    unsigned& height,
    unsigned& numberOfChannels,
    unsigned& bitsPerChannel
);

bmpStatus_t LoadBMP(
	void**				data,
	BaseAllocator		*alloc,
    unsigned            &surfaceHeight,
    unsigned            &surfaceWidth,
    unsigned            &surfacePitch8,
	unsigned			&channels,
    const char *filename
);

bmpStatus_t StoreBMP(
    const char           *filename,
	unsigned char 		 *h_Surface,
	unsigned			  channels,
    unsigned              surfaceHeight,
    unsigned              surfaceWidth,
    unsigned              surfacePitch8
);

#endif
