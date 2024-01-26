
#ifndef __AFFINE_OPTIONS__
#define __AFFINE_OPTIONS__

#include "fastvideo_sdk_define.h"

class AffineOptions {
private:
	bool AffineTypeParser( const char *affineType );

public:
	fastAffineTransformations_t AffineType;

	AffineOptions(void) { };
	~AffineOptions(void) { };

	bool Parse(int argc, char *argv[]);
};

#endif // __AFFINE_OPTIONS__
