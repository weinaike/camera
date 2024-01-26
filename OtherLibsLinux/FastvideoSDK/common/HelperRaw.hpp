#pragma once

#include "fastvideo_sdk_define.h"
#include "alignment.hpp"

inline unsigned long GetHostRawPitch(fastRawFormat_t rawFmt, unsigned width) {
	switch (rawFmt)
	{
		case FAST_RAW_XIMEA12: 
			return uSnapUp((unsigned)width, (unsigned)2) * 1.5f;
		case FAST_RAW_PTG12:
			return uSnapUp((unsigned)width, (unsigned)2) * 1.5f;
	};
}

inline unsigned long GetDeviceRawPitch(fastRawFormat_t rawFmt, unsigned width) {
	switch (rawFmt)
	{
		case FAST_RAW_XIMEA12:
			return uSnapUp((unsigned)width, (unsigned)8) * 1.5f;
		case FAST_RAW_PTG12:
			return uSnapUp((unsigned)width, (unsigned)8) * 1.5f;
	};
}
