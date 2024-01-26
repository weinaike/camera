
#ifndef __NPP_RESIZE_OPTIONS__
#define __NPP_RESIZE_OPTIONS__

#include "fastvideo_nppResize.h"

class NppResizeOptions {
public:
	NppResizeOptions(void) {};
	~NppResizeOptions(void) {};
	unsigned ResizedWidth;
	unsigned ResizedHeight;
	bool ResizedHeightEnabled;

	double Shift;

	bool Parse(int argc, char *argv[]);
};

#endif // __NPP_RESIZE_OPTIONS__
