
#ifndef __NPP_ROTATE_OPTIONS__
#define __NPP_ROTATE_OPTIONS__

#include "fastvideo_nppRotate.h"
#include "InterpolationOptions.h"

class NppRotateOptions {
public:
	NppRotateOptions(void) {};
	~NppRotateOptions(void) {};

	
	double Angle;

	double Shift;

	bool Parse(int argc, char *argv[]);
};

#endif // __NPP_ROTATE_OPTIONS__
