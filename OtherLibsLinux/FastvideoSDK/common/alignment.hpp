
#ifndef __ALIGNMENT_HPP__
#define __ALIGNMENT_HPP__

////////////////////////////////////////////////////////////////////////////////
// Common math
////////////////////////////////////////////////////////////////////////////////
template<typename T> static inline T uDivUp(T a, T b){
    return (a / b) + (a % b != 0);
}

template<typename T> static inline T uSnapUp(T a, T b){
    return ( a + (b - a % b) % b );
}

template<typename T> static inline T uSnapDown(T a, T b) {
	return (a - (a % b));
}

template<typename T> static inline T uSnapDelta(T a, T b){
    return (b - a % b) % b;
}

unsigned GetAlignedPitch(unsigned width, unsigned channels , unsigned bytePerChannel, unsigned boundary);
unsigned GetAlignedPitch(unsigned width, unsigned channels, float bytePerChannel, unsigned boundary);

#endif
