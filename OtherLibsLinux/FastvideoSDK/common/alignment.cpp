#include "cmath"

#include "alignment.hpp"

unsigned GetAlignedPitch(unsigned width, unsigned channels, unsigned bytePerChannel, unsigned boundary) {
	return uSnapUp(width * channels * bytePerChannel, boundary);
}

unsigned GetAlignedPitch(unsigned width, unsigned channels, float bytePerChannel, unsigned boundary) {
	return uSnapUp(static_cast<unsigned>(ceilf(width * channels * bytePerChannel)), boundary);
}
