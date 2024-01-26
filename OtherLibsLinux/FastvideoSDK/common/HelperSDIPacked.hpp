#pragma once

inline unsigned AlignSDIValuePacked(unsigned value) {
	return uDivUp(value, 6u);
}

inline unsigned long GetSDIWidth(fastSDIFormat_t sdiFmt, unsigned width) {
	switch (sdiFmt) {
		case FAST_SDI_422_8_CbYCrY_BT601:
		case FAST_SDI_422_8_CbYCrY_BT601_FR:
		case FAST_SDI_422_8_CbYCrY_BT709:
		case FAST_SDI_422_8_CbYCrY_BT2020:

		case FAST_SDI_422_8_CrYCbY_BT601:
		case FAST_SDI_422_8_CrYCbY_BT601_FR:
		case FAST_SDI_422_8_CrYCbY_BT709:
		case FAST_SDI_422_8_CrYCbY_BT2020:
		
		case FAST_SDI_422_10_CbYCrY_PACKED_BT601:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT601_FR:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT709:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT2020:
		{
			return width * 2;
		}

		case FAST_SDI_RGB_10_BMR10L:
		case FAST_SDI_RGB_10_BMR10B:
		{
			return width;
		}

		case FAST_SDI_RGB_12_BMR12B:
		case FAST_SDI_RGB_12_BMR12L:
		{
			return uSnapUp(width, 8u);
		}
	}

	return 0;
}

inline unsigned long GetSDIHeight(fastSDIFormat_t sdiFmt, unsigned height) {
	switch (sdiFmt) {
		case FAST_SDI_422_8_CbYCrY_BT601:
		case FAST_SDI_422_8_CbYCrY_BT601_FR:
		case FAST_SDI_422_8_CbYCrY_BT709:
		case FAST_SDI_422_8_CbYCrY_BT2020:

		case FAST_SDI_422_8_CrYCbY_BT601:
		case FAST_SDI_422_8_CrYCbY_BT601_FR:
		case FAST_SDI_422_8_CrYCbY_BT709:
		case FAST_SDI_422_8_CrYCbY_BT2020:

		case FAST_SDI_422_10_CbYCrY_PACKED_BT601:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT601_FR:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT709:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT2020:

		case FAST_SDI_RGB_10_BMR10L:
		case FAST_SDI_RGB_10_BMR10B:
		case FAST_SDI_RGB_12_BMR12B:
		case FAST_SDI_RGB_12_BMR12L:
		{
			return height;
		}
	}

	return 0;
}

inline unsigned long GetSDIRowSizeInByte(fastSDIFormat_t sdiFmt, unsigned width) {
	switch (sdiFmt) {
		case FAST_SDI_422_8_CbYCrY_BT601:
		case FAST_SDI_422_8_CbYCrY_BT601_FR:
		case FAST_SDI_422_8_CbYCrY_BT709:
		case FAST_SDI_422_8_CbYCrY_BT2020:

		case FAST_SDI_422_8_CrYCbY_BT601:
		case FAST_SDI_422_8_CrYCbY_BT601_FR:
		case FAST_SDI_422_8_CrYCbY_BT709:
		case FAST_SDI_422_8_CrYCbY_BT2020:
		{
			return GetSDIWidth(sdiFmt, width) * GetSDIValueSize(sdiFmt);
		}

		case FAST_SDI_422_10_CbYCrY_PACKED_BT601:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT601_FR:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT709:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT2020:
		{
			return GetSDIWidth(sdiFmt, width) * sizeof(unsigned) / 3;
		}

		case FAST_SDI_RGB_10_BMR10L:
		case FAST_SDI_RGB_10_BMR10B:
		{
			return GetSDIWidth(sdiFmt, width) * sizeof(unsigned);
		}

		case FAST_SDI_RGB_12_BMR12B:
		case FAST_SDI_RGB_12_BMR12L:
		{
			return (GetSDIWidth(sdiFmt, width) / 8) * 9 * sizeof(unsigned);
		}
	}
	return 0;
}

inline unsigned long GetSDIPitchCommon(fastSDIFormat_t sdiFmt, unsigned width, bool isHost) {
	return AlignSDIPitch(GetSDIRowSizeInByte(sdiFmt, width), isHost); 
}

inline unsigned long GetSDIBufferSizeCommonPacked(fastSDIFormat_t sdiFmt, unsigned width, unsigned height, bool isHost) {
	switch (sdiFmt) {
		case FAST_SDI_422_8_CbYCrY_BT601:
		case FAST_SDI_422_8_CbYCrY_BT601_FR:
		case FAST_SDI_422_8_CbYCrY_BT709:
		case FAST_SDI_422_8_CbYCrY_BT2020:

		case FAST_SDI_422_8_CrYCbY_BT601:
		case FAST_SDI_422_8_CrYCbY_BT601_FR:
		case FAST_SDI_422_8_CrYCbY_BT709:
		case FAST_SDI_422_8_CrYCbY_BT2020:

		case FAST_SDI_422_10_CbYCrY_PACKED_BT601:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT601_FR:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT709:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT2020:

		case FAST_SDI_RGB_10_BMR10L:
		case FAST_SDI_RGB_10_BMR10B:
		case FAST_SDI_RGB_12_BMR12B:
		case FAST_SDI_RGB_12_BMR12L:
		{
			return GetSDIPitchCommon(sdiFmt, width, isHost) * GetSDIHeight(sdiFmt, height);
		}
	}
	return 0;
}

/////////////////////////////////////////////////
/// SDI helpers (HOST version)
/////////////////////////////////////////////////

inline unsigned long GetSDIPitch(fastSDIFormat_t sdiFmt, unsigned width) {
	return GetSDIPitchCommon(sdiFmt, width, true);
}

/////////////////////////////////////////////////
/// SDI helpers (DEVICE version)
/////////////////////////////////////////////////

inline unsigned long GetDeviceSDIPitch(fastSDIFormat_t sdiFmt, unsigned width) {
	return GetSDIPitchCommon(sdiFmt, width, false);
}

/////////////////////////////////////////////////
/// SDI pitch converter (DEVICE <-> HOST)
///		Note: device memory aligned to FAST_ALIGNMENT
///			  host memory not aligned
/////////////////////////////////////////////////

inline void PackDeviceSDIPacked(
	unsigned char *src,
	unsigned char *dst,

	fastSDIFormat_t sdiFmt,
	unsigned width, unsigned height
) {
	switch (sdiFmt) {
		case FAST_SDI_422_8_CbYCrY_BT601:
		case FAST_SDI_422_8_CbYCrY_BT601_FR:
		case FAST_SDI_422_8_CbYCrY_BT709:
		case FAST_SDI_422_8_CbYCrY_BT2020:

		case FAST_SDI_422_8_CrYCbY_BT601:
		case FAST_SDI_422_8_CrYCbY_BT601_FR:
		case FAST_SDI_422_8_CrYCbY_BT709:
		case FAST_SDI_422_8_CrYCbY_BT2020:
		
		case FAST_SDI_422_10_CbYCrY_PACKED_BT601_FR:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT601:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT709:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT2020:
		{
			// just copy data to output
			memcpy(dst, src, GetSDIValueSize(sdiFmt) * GetSDIWidth(sdiFmt, width) * GetSDIHeight(sdiFmt, height));
			break;
		}
	}
}

inline void UnPackDeviceSDIPacked(unsigned char *src,
	unsigned char *dst,

	fastSDIFormat_t sdiFmt,
	unsigned width, unsigned height
) {
	switch (sdiFmt) {
		case FAST_SDI_422_8_CbYCrY_BT601:
		case FAST_SDI_422_8_CbYCrY_BT601_FR:
		case FAST_SDI_422_8_CbYCrY_BT709:
		case FAST_SDI_422_8_CbYCrY_BT2020:

		case FAST_SDI_422_8_CrYCbY_BT601:
		case FAST_SDI_422_8_CrYCbY_BT601_FR:
		case FAST_SDI_422_8_CrYCbY_BT709:
		case FAST_SDI_422_8_CrYCbY_BT2020:
		
		case FAST_SDI_422_10_CbYCrY_PACKED_BT601_FR:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT601:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT709:
		case FAST_SDI_422_10_CbYCrY_PACKED_BT2020:

		{
			// just copy data to output
			memcpy(dst, src, GetSDIValueSize(sdiFmt) * GetSDIWidth(sdiFmt, width) * GetSDIHeight(sdiFmt, height));
			break;
		}
	}
}
