#ifndef __SURFACE_TRAITS__
#define __SURFACE_TRAITS__

#include "fast_enums.h"
#include "alignment.hpp"

template<fastSurfaceFormat_t kSurface> struct SurfaceTrait {
public:
	typedef unsigned char Element;
	static const fastSurfaceFormat_t SurfaceFmt;
	static const unsigned NumberOfChannels = 0;
	static const unsigned BytesPerChannel = 0;
	static const unsigned BitsPerChannel = 0;

	static unsigned Pitch(const unsigned width) {
		return 0;
	}

	static unsigned GetBytesPerChannel(void) {
		return 0;
	}
};

#include "SurfaceTraitsInternal.hpp"

/////////////////////////////////////////////////////////////////////////////////
// Grayscale surface parameters
/////////////////////////////////////////////////////////////////////////////////

template<> struct SurfaceTrait<FAST_I8> {
public:
	typedef unsigned char Element;
	static const fastSurfaceFormat_t SurfaceFmt = FAST_I8;
	static const unsigned NumberOfChannels = 1;
	static const unsigned BytesPerChannel = 1;
	static const unsigned BitsPerChannel = 8;

	static unsigned Pitch(const unsigned width) {
		return GetAlignedPitch(width, NumberOfChannels, BytesPerChannel, FAST_ALIGNMENT);
	}

	static unsigned GetBytesPerChannel(void) {
		return BytesPerChannel;
	}
};

template<> struct SurfaceTrait<FAST_I12> {
public:
	typedef unsigned short Element;
	static const fastSurfaceFormat_t SurfaceFmt = FAST_I12;
	static const unsigned NumberOfChannels = 1;
	static const unsigned BytesPerChannel = 2;
	static const unsigned BitsPerChannel = 12;

	static unsigned Pitch(const unsigned width) {
		return GetAlignedPitch(width, NumberOfChannels, BytesPerChannel, FAST_ALIGNMENT);
	}

	static unsigned GetBytesPerChannel(void) {
		return BytesPerChannel;
	}
};

template<> struct SurfaceTrait<FAST_I16> {
public:
	typedef unsigned short Element;
	static const fastSurfaceFormat_t SurfaceFmt = FAST_I16;
	static const unsigned NumberOfChannels = 1;
	static const unsigned BytesPerChannel = 2;
	static const unsigned BitsPerChannel = 16;

	static unsigned Pitch(const unsigned width) {
		return GetAlignedPitch(width, NumberOfChannels, BytesPerChannel, FAST_ALIGNMENT);
	}

	static unsigned GetBytesPerChannel(void) {
		return BytesPerChannel;
	}
};

template<> struct SurfaceTrait<FAST_I10> {
public:
	typedef unsigned short Element;
	static const fastSurfaceFormat_t SurfaceFmt = FAST_I10;
	static const unsigned NumberOfChannels = 1;
	static const unsigned BytesPerChannel = 2;
	static const unsigned BitsPerChannel = 10;

	static unsigned Pitch(const unsigned width) {
		return GetAlignedPitch(width, NumberOfChannels, BytesPerChannel, FAST_ALIGNMENT);
	}

	static unsigned GetBytesPerChannel(void) {
		return BytesPerChannel;
	}
};

template<> struct SurfaceTrait<FAST_I14> {
public:
	typedef unsigned short Element;
	static const fastSurfaceFormat_t SurfaceFmt = FAST_I14;
	static const unsigned NumberOfChannels = 1;
	static const unsigned BytesPerChannel = 2;
	static const unsigned BitsPerChannel = 14;

	static unsigned Pitch(const unsigned width) {
		return GetAlignedPitch(width, NumberOfChannels, BytesPerChannel, FAST_ALIGNMENT);
	}

	static unsigned GetBytesPerChannel(void) {
		return BytesPerChannel;
	}
};

/////////////////////////////////////////////////////////////////////////////////
// Color surface parameters
/////////////////////////////////////////////////////////////////////////////////

template<> struct SurfaceTrait<FAST_RGB8> {
public:
	typedef unsigned char Element;
	static const fastSurfaceFormat_t SurfaceFmt = FAST_RGB8;
	static const unsigned NumberOfChannels = 3;
	static const unsigned BytesPerChannel = 1;
	static const unsigned BitsPerChannel = 8;

	static unsigned Pitch(const unsigned width) {
		return GetAlignedPitch(width, NumberOfChannels, BytesPerChannel, FAST_ALIGNMENT);
	}

	static unsigned GetBytesPerChannel(void) {
		return BytesPerChannel;
	}
};

template<> struct SurfaceTrait<FAST_BGRX8> {
public:
	typedef unsigned char Element;
	static const fastSurfaceFormat_t SurfaceFmt = FAST_BGRX8;
	static const unsigned NumberOfChannels = 4;
	static const unsigned BytesPerChannel = 1;
	static const unsigned BitsPerChannel = 8;

	static unsigned Pitch(const unsigned width) {
		return GetAlignedPitch(width, NumberOfChannels, BytesPerChannel, FAST_ALIGNMENT);
	}

	static unsigned GetBytesPerChannel(void) {
		return BytesPerChannel;
	}
};

template<> struct SurfaceTrait<FAST_BGR8> {
public:
	typedef unsigned char Element;
	static const fastSurfaceFormat_t SurfaceFmt = FAST_BGR8;
	static const unsigned NumberOfChannels = 3;
	static const unsigned BytesPerChannel = 1;
	static const unsigned BitsPerChannel = 8;

	static unsigned Pitch(const unsigned width) {
		return GetAlignedPitch(width, NumberOfChannels, BytesPerChannel, FAST_ALIGNMENT);
	}

	static unsigned GetBytesPerChannel(void) {
		return BytesPerChannel;
	}
};

template<> struct SurfaceTrait<FAST_RGB12> {
public:
	typedef unsigned short Element;
	static const fastSurfaceFormat_t SurfaceFmt = FAST_RGB12;
	static const unsigned NumberOfChannels = 3;
	static const unsigned BytesPerChannel = 2;
	static const unsigned BitsPerChannel = 12;

	static unsigned Pitch(const unsigned width) {
		return GetAlignedPitch(width, NumberOfChannels, BytesPerChannel, FAST_ALIGNMENT);
	}

	static unsigned GetBytesPerChannel(void) {
		return BytesPerChannel;
	}
};

template<> struct SurfaceTrait<FAST_RGB16> {
public:
	typedef unsigned short Element;
	static const fastSurfaceFormat_t SurfaceFmt = FAST_RGB16;
	static const unsigned NumberOfChannels = 3;
	static const unsigned BytesPerChannel = 2;
	static const unsigned BitsPerChannel = 16;

	static unsigned Pitch(const unsigned width) {
		return GetAlignedPitch(width, NumberOfChannels, BytesPerChannel, FAST_ALIGNMENT);
	}

	static unsigned GetBytesPerChannel(void) {
		return BytesPerChannel;
	}
};


size_t GetPitchFromSurface(const fastSurfaceFormat_t surfaceFmt, const unsigned width);
unsigned GetBytesPerChannelFromSurface(const fastSurfaceFormat_t surfaceFmt);
unsigned GetNumberOfChannelsFromSurface(const fastSurfaceFormat_t surfaceFmt);
unsigned GetBitsPerChannelFromSurface(const fastSurfaceFormat_t surfaceFmt);
size_t GetBufferSizeFromSurface(const fastSurfaceFormat_t surfaceFmt, const unsigned width, const unsigned height);

#endif //__SURFACE_TRAITS__
