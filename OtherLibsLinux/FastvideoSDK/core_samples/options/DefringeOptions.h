
#ifndef __DEFRINGE_OPTIONS__
#define __DEFRINGE_OPTIONS__

#include "fastvideo_sdk_define.h"

class DefringeOptions {
public:
	unsigned WindowSize;

	unsigned short TintR;
	unsigned short TintG;
	unsigned short TintB;

	float Fi_tint; // ����, ������������ �������: [-180, +180]
	float Fi_max; // ������������ ���� ���������� �� ��������� �������: [60, 90]
	float Coefficient; // [0.001, 0.1]

	DefringeOptions(void) { };
	~DefringeOptions(void) { };

	bool Parse(int argc, char *argv[]);
};

#endif // __DEFRINGE_OPTIONS__
