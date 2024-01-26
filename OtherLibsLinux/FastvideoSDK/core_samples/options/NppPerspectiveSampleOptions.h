
#ifndef __NPP_PERSPECTIVE_SAMPLE_OPTIONS__
#define __NPP_PERSPECTIVE_SAMPLE_OPTIONS__

#include "BaseOptions.h"
#include "InterpolationOptions.h"
#include "NppPerspectiveOptions.h"

class NppPerspectiveSampleOptions : public virtual BaseOptions {
public:
	NppPerspectiveSampleOptions() = default;
	~NppPerspectiveSampleOptions() = default;

	NppInterpolationOptions Interpolation;
	NppPerspectiveOptions Perspective;

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __NPP_PERSPECTIVE_SAMPLE_OPTIONS__
