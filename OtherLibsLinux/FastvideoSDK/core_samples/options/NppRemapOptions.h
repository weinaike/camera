
#ifndef __NPP_REMAP_OPTIONS__
#define __NPP_REMAP_OPTIONS__

#include "fastvideo_nppGeometry.h"

class NppRemapOptions {
public:
	bool Rotate90;

	unsigned short BackgroundR;
	unsigned short BackgroundG;
	unsigned short BackgroundB;
	bool BackgroundEnabled;

	fastNppGeometryTransformationType_t TransformType;

	NppRemapOptions(void) {};
	~NppRemapOptions(void) {};

	bool Parse(int argc, char *argv[]);
};

#endif // __NPP_REMAP_OPTIONS__
