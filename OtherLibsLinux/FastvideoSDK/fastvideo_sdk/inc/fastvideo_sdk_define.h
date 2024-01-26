
#ifndef __FAST_DEFINE_H__
#define __FAST_DEFINE_H__

#include <stdlib.h>

#ifdef __cplusplus
    extern "C" {
#endif

#ifndef __GNUC__

#ifdef FAST_EXPORTS
	#define DLL __declspec(dllexport) __cdecl
#else
	#define DLL
#endif

#else

#define DLL

#endif

////////////////////////////////////////////////////////////////////////////////
// Basic data types
////////////////////////////////////////////////////////////////////////////////
#define FAST_ALIGNMENT 4U
#define FAST_SCALE_FACTOR_MAX 40U
#define FAST_MIN_SCALED_SIZE 32U

typedef enum {
	FAST_OK,                           // There is no error during function execution

	FAST_TRIAL_PERIOD_EXPIRED,

	FAST_INVALID_DEVICE,               // Device with selected index does not exist or device is non NVIDIA device or device is non CUDA-compatible device
	FAST_INCOMPATIBLE_DEVICE,          // Device is CUDA-compatible, but its compute compatibility is below 2.0, thus device is considered to be incompatible with SDK
	FAST_INSUFFICIENT_DEVICE_MEMORY,   // Available device memory is not enough to allocate new buffer
	FAST_INSUFFICIENT_HOST_MEMORY,     // Available host memory is not enough to allocate new buffer
	FAST_INVALID_HANDLE,               // Component handle is invalid or has inappropriate type
	FAST_INVALID_VALUE,                // Some parameter of the function called is invalid or combination of input parameters are unacceptable
	FAST_UNAPPLICABLE_OPERATION,	   // This operation can not be applied to the current type of data
	FAST_INVALID_SIZE,                 // Image dimension is invalid
	FAST_UNALIGNED_DATA,               // Buffer base pointers or pitch are not properly aligned
	FAST_INVALID_TABLE,                // Invalid quantization / Huffman table
	FAST_BITSTREAM_CORRUPT,            // JPEG bitstream is corrupted and can not be decoded
	FAST_EXECUTION_FAILURE,            // Device kernel execution failure
	FAST_INTERNAL_ERROR,               // Internal error, non-kernel software execution failure
	FAST_UNSUPPORTED_SURFACE,

	FAST_IO_ERROR,                     // Failed to read/write file
	FAST_INVALID_FORMAT,               // Invalid file format
	FAST_UNSUPPORTED_FORMAT,           // File format is not supported by the current version of SDK
	FAST_END_OF_STREAM,

	FAST_MJPEG_THREAD_ERROR,
	FAST_TIMEOUT,
	FAST_MJPEG_OPEN_FILE_ERROR,

	FAST_UNKNOWN_ERROR,                 // Unrecognized error
	FAST_INCOMPATIBLE_DRIVER
} fastStatus_t;

typedef enum {
	FAST_I8,
	FAST_I10,
	FAST_I12,
	FAST_I14,
	FAST_I16,

	FAST_RGB8,
	FAST_BGR8,
	FAST_RGB12,
	FAST_RGB16,

	FAST_BGRX8,

	FAST_CrCbY8,
	FAST_YCbCr8
} fastSurfaceFormat_t;

typedef enum {
	FAST_JPEG_SEQUENTIAL_DCT, FAST_JPEG_LOSSLESS
} fastJpegMode_t;

typedef enum {
	FAST_JPEG_Y, FAST_JPEG_444, FAST_JPEG_422, FAST_JPEG_420
} fastJpegFormat_t;

typedef enum {
	FAST_BAYER_NONE,
	FAST_BAYER_RGGB,
	FAST_BAYER_BGGR,
	FAST_BAYER_GBRG,
	FAST_BAYER_GRBG,
} fastBayerPattern_t;

typedef enum {
	FAST_DFPD,
	FAST_HQLI,
	FAST_MG,
	FAST_MG2,
	FAST_MG_V2,

	FAST_BINNING_2x2,
	FAST_BINNING_4x4,
	FAST_BINNING_8x8,
	FAST_L7,
	FAST_AMAZE
} fastDebayerType_t;

typedef enum {
	FAST_RAW_XIMEA12,
	FAST_RAW_PTG12,
} fastRawFormat_t;

typedef enum {
	FAST_SDI_YV12_BT601_FR,
	FAST_SDI_YV12_BT601,
	FAST_SDI_YV12_BT709,
	FAST_SDI_YV12_BT2020,

	FAST_SDI_NV12_BT601_FR,
	FAST_SDI_NV12_BT601,
	FAST_SDI_NV12_BT709,
	FAST_SDI_NV12_BT2020,

	FAST_SDI_P010_BT601_FR,
	FAST_SDI_P010_BT601,
	FAST_SDI_P010_BT709,
	FAST_SDI_P010_BT2020,

	FAST_SDI_420_8_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_420_8_YCbCr_PLANAR_BT601,
	FAST_SDI_420_8_YCbCr_PLANAR_BT709,
	FAST_SDI_420_8_YCbCr_PLANAR_BT2020,

	FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT601,
	FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT709,
	FAST_SDI_420_10_LSBZ_YCbCr_PLANAR_BT2020,

	FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT601,
	FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT709,
	FAST_SDI_420_10_MSBZ_YCbCr_PLANAR_BT2020,

	FAST_SDI_422_8_CbYCrY_BT601_FR,
	FAST_SDI_422_8_CbYCrY_BT601,
	FAST_SDI_422_8_CbYCrY_BT709,
	FAST_SDI_422_8_CbYCrY_BT2020,

	FAST_SDI_422_8_CrYCbY_BT601_FR,
	FAST_SDI_422_8_CrYCbY_BT601,
	FAST_SDI_422_8_CrYCbY_BT709,
	FAST_SDI_422_8_CrYCbY_BT2020,

	FAST_SDI_422_10_CbYCrY_PACKED_BT2020,
	FAST_SDI_422_10_CbYCrY_PACKED_BT601_FR,
	FAST_SDI_422_10_CbYCrY_PACKED_BT601,
	FAST_SDI_422_10_CbYCrY_PACKED_BT709,

	FAST_SDI_422_8_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_422_8_YCbCr_PLANAR_BT601,
	FAST_SDI_422_8_YCbCr_PLANAR_BT709,
	FAST_SDI_422_8_YCbCr_PLANAR_BT2020,

	FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT601,
	FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT709,
	FAST_SDI_422_10_MSBZ_YCbCr_PLANAR_BT2020,

	FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT601,
	FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT709,
	FAST_SDI_422_10_LSBZ_YCbCr_PLANAR_BT2020,

	FAST_SDI_444_8_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_444_8_YCbCr_PLANAR_BT601,
	FAST_SDI_444_8_YCbCr_PLANAR_BT709,
	FAST_SDI_444_8_YCbCr_PLANAR_BT2020,

	FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601_FR,
	FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT601,
	FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT709,
	FAST_SDI_444_10_LSBZ_YCbCr_PLANAR_BT2020,

	FAST_SDI_RGBA,

	FAST_SDI_RGB_10_BMR10L,
	FAST_SDI_RGB_12_BMR12B,
	FAST_SDI_RGB_12_BMR12L,
	FAST_SDI_RGB_10_BMR10B

} fastSDIFormat_t;

typedef enum {
	FAST_RGBA_ALPHA_PADDING_ZERO,
	FAST_RGBA_ALPHA_PADDING_FF,
} fastRGBAAlphaPadding_t;

typedef enum {
	FAST_CHANNEL_H,
	FAST_CHANNEL_S,
	FAST_CHANNEL_L_OR_V
} fastColorSaturationChannelType_t;

typedef enum {
	FAST_CHANNEL_R,
	FAST_CHANNEL_G,
	FAST_CHANNEL_B
} fastChannelType_t;

typedef enum {
	FAST_OP_REPLACE,
	FAST_OP_ADD,
	FAST_OP_MULTIPLY
} fastColorSaturationOperationType_t;

typedef enum {
	FAST_BINNING_NONE,
	FAST_BINNING_SUM,
	FAST_BINNING_AVERAGE
} fastBinningMode_t;

typedef enum {
	FAST_GAUSSIAN_SHARPEN = 1,
	FAST_LUT_8_8,
	FAST_LUT_8_8_C,
	FAST_LUT_8_12,
	FAST_LUT_8_12_C,
	FAST_LUT_8_16,
	FAST_LUT_8_16_C,
	FAST_LUT_8_16_BAYER,

	FAST_LUT_10_16,
	FAST_LUT_10_16_BAYER,

	FAST_LUT_12_8,
	FAST_LUT_12_8_C,
	FAST_LUT_12_12,
	FAST_LUT_12_12_C,
	FAST_LUT_12_16,
	FAST_LUT_12_16_C,
	FAST_LUT_12_16_BAYER,

	FAST_LUT_14_16,
	FAST_LUT_14_16_BAYER,

	FAST_LUT_16_16,
	FAST_LUT_16_16_C,
	FAST_LUT_16_8,
	FAST_LUT_16_8_C,
	FAST_LUT_16_16_BAYER,

	FAST_LUT_16_16_FR,
	FAST_LUT_16_16_FR_C,
	FAST_LUT_16_16_FR_BAYER,

	FAST_HSV_LUT_3D,
	FAST_RGB_LUT_3D,

	FAST_TONE_CURVE,
	
	FAST_SAM,
	FAST_SAM16,

	FAST_BASE_COLOR_CORRECTION,
	FAST_WHITE_BALANCE,
	FAST_COLUMN_FILTER,
	FAST_COLOR_SATURATION_HSL,
	FAST_COLOR_SATURATION_HSV,

	FAST_MEDIAN,
	FAST_BAYER_BLACK_SHIFT,
	FAST_DEFRINGE,
	FAST_BAD_PIXEL_CORRECTION_5X5,
	FAST_BINNING,
	FAST_GAUSSIAN_BLUR,
	FAST_FFC
} fastImageFilterType_t;

typedef enum {
	FAST_BIT_DEPTH,
	FAST_SELECT_CHANNEL,
	FAST_RGB_TO_GRAYSCALE,
	FAST_GRAYSCALE_TO_GRAYSCALERGB,
	FAST_BAYER_TO_RGB,
	FAST_GRAYSCALE_TO_RGB
} fastSurfaceConverter_t;

typedef enum {
	FAST_AFFINE_NOP = 0,
	FAST_AFFINE_FLIP = 1,
	FAST_AFFINE_FLOP = 2,
	FAST_AFFINE_ROTATION180 = 4,
	FAST_AFFINE_ROTATION90LEFT = 8,
	FAST_AFFINE_ROTATION90LEFT_FLOPPED = 16,
	FAST_AFFINE_ROTATION90RIGHT = 32,
	FAST_AFFINE_ROTATION90RIGHT_FLOPPED = 64,
	FAST_AFFINE_ALL = 127
} fastAffineTransformations_t;

typedef enum {
	FAST_LANCZOS = 1
} fastResizeType_t;

typedef enum {
	FAST_CONVERT_NONE,
	FAST_CONVERT_BGR
} fastConvertType_t;

typedef enum {
	FAST_HISTOGRAM_COMMON,
	FAST_HISTOGRAM_BAYER,
	FAST_HISTOGRAM_BAYER_G1G2,
	FAST_HISTOGRAM_PARADE,
} fastHistogramType_t;

typedef enum {
	FAST_HDR_IMPORT_16,
	FAST_HDR_IMPORT_32
} fastHDRImportType_t;

typedef enum {
	FAST_HDR_GRAY_3x12
} fastHDRBuilderFormat_t;

typedef enum {
	FAST_HDR_EXPOSURE_X1,
	FAST_HDR_EXPOSURE_X4,
	FAST_HDR_EXPOSURE_X16
} fastHDRImageExposure_t;

typedef enum {
	FAST_LICENSE_TYPE_DEMO,
	FAST_LICENSE_TYPE_TRIAL,
	FAST_LICENSE_TYPE_STANDARD_SENSELOCK,
	FAST_LICENSE_TYPE_STANDARD_GUARDANT,
	FAST_LICENSE_TYPE_ENTERPRISE
} fastLicenseType_t;

typedef enum {
	FAST_LICENSE_PROVIDER_NONE,
	FAST_LICENSE_PROVIDER_SENSELOCK_DONGLE,
	FAST_LICENSE_PROVIDER_GUARDANT_DONGLE,
	FAST_LICENSE_PROVIDER_GUARDANT_SOFT_KEY
} fastLicenseProvider_t;

typedef enum {
	FAST_IMPORT_FROM_HOST,
	FAST_IMPORT_FROM_DEVICE,
	FAST_RAW_IMPORT_FROM_HOST,
	FAST_RAW_IMPORT_FROM_DEVICE,
	FAST_SDI_IMPORT_FROM_HOST,
	FAST_SDI_IMPORT_FROM_DEVICE,

	FAST_EXPORT_TO_HOST,
	FAST_EXPORT_TO_DEVICE,
	FAST_RAW_EXPORT_TO_HOST,
	FAST_RAW_EXPORT_TO_DEVICE,
	FAST_SDI_EXPORT_TO_HOST,
	FAST_SDI_EXPORT_TO_DEVICE,

	FAST_JPEG_ENCODER,
	FAST_JPEG_DECODER,
	FAST_DEBAYER,
	FAST_RESIZE,
	FAST_FILTER,
	FAST_CROP,
	FAST_AFFINE,
	FAST_SURFACE_CONVERTER,
	FAST_DENOISE
} fastComponentType_t;

typedef enum {
	FAST_RESIZE_PRESERV_ASPECT_RATIO,
	FAST_RESIZE_NO_PRESERV_ASPECT_RATIO,
	FAST_RESIZE_PADDING_BACKGROUND
} fastResizeMethod_t;

typedef enum {
	FAST_THRESHOLD_FUNCTION_UNKNOWN,
	FAST_THRESHOLD_FUNCTION_HARD,
	FAST_THRESHOLD_FUNCTION_SOFT,
	FAST_THRESHOLD_FUNCTION_GARROTE
} fastDenoiseThresholdFunctionType_t;

typedef enum {
	FAST_WAVELET_CDF97,
	FAST_WAVELET_CDF53
} fastWaveletType_t;

////////////////////////////////////////////////////////////////////////////////
// 
////////////////////////////////////////////////////////////////////////////////
typedef struct fastEncoderHandleStruct *fastJpegEncoderHandle_t;
typedef struct fastDecoderHandleStruct *fastJpegDecoderHandle_t;

typedef struct fastEncoderHandleStruct_v2 *fastJpegEncoderHandle_v2_t;
typedef struct fastDecoderHandleStruct_v2 *fastJpegDecoderHandle_v2_t;

typedef struct fastDebayerHandleStruct *fastDebayerHandle_t;
typedef struct fastResizerHandleStruct *fastResizerHandle_t;
typedef struct fastSurfaceConverterHandleStruct *fastSurfaceConverterHandle_t;
typedef struct fastImageFiltersHandleStruct *fastImageFiltersHandle_t;
typedef struct fastCropHandleStruct *fastCropHandle_t;
typedef struct fastHDRImportStruct* fastHDRImportHandle_t;
typedef struct fastHdrBuilderStruct* fastHDRBuilderHandle_t;
typedef struct fastGpuTimerStruct *fastGpuTimerHandle_t;
typedef struct fastExportToHostStruct *fastExportToHostHandle_t;
typedef struct fastExportToDeviceStruct *fastExportToDeviceHandle_t;
typedef struct fastImportFromHostStruct *fastImportFromHostHandle_t;
typedef struct fastImportFromDeviceStruct *fastImportFromDeviceHandle_t;
typedef struct fastExternalBufferWrapperStruct*fastExternalBufferWrapperHandle_t;
typedef struct fastAffineHandleStruct *fastAffineHandle_t;
typedef struct fastSimplePipelineHandleStruct* fastSimplePipelineHandle_t;
typedef struct fastRawUnpackerHandleStruct *fastRawUnpackerHandle_t;
typedef struct fastBayerMergerHandleStruct *fastBayerMergerHandle_t;
typedef struct fastBayerSplitterHandleStruct *fastBayerSplitterHandle_t;
typedef struct fastMuxStruct *fastMuxHandle_t;
typedef struct fastSDIImportFromHostHandleStruct *fastSDIImportFromHostHandle_t;
typedef struct fastSDIImportFromDeviceHandleStruct *fastSDIImportFromDeviceHandle_t;
typedef struct fastSDIExportToHostHandleStruct *fastSDIExportToHostHandle_t;
typedef struct fastSDIExportToDeviceHandleStruct *fastSDIExportToDeviceHandle_t;
typedef struct fastHistogramHandleStruct *fastHistogramHandle_t;
typedef struct fastDenoiseHandleStruct* fastDenoiseHandle_t;

typedef struct fastSdkParameters *fastSdkParametersHandle_t;
typedef struct fastSimplePipelineHandleStruct* fastSimplePipelineHandle_t;

typedef struct fastDeviceSurfaceBuffer_t *fastDeviceSurfaceBufferHandle_t;

typedef struct {
	unsigned short R;
	unsigned short G;
	unsigned short B;
} fastRgb_t;

typedef struct {
	float H;
	float S;
	float V;
} fastHSVfloat_t;

typedef struct{
	unsigned short data[64];
} fastQuantTable_t;

typedef struct {
	fastQuantTable_t table[4];
} fastJpegQuantState_t;

typedef struct {
	unsigned char bucket[16];
	unsigned char alphabet[256];
} fastHuffmanTable_t;

typedef struct {
	fastHuffmanTable_t table[2][2];
} fastJpegHuffmanState_t;

typedef struct {
	unsigned quantTableMask;
	unsigned huffmanTableMask[2];
	unsigned scanChannelMask;
	unsigned scanGroupMask;
} fastJpegScanStruct_t;

typedef struct {
	unsigned short exifCode;
	char *exifData;
	int exifLength;
} fastJpegExifSection_t;

typedef struct {
	fastJpegMode_t jpegMode;
	fastJpegFormat_t jpegFmt;

	int predictorClass;

	unsigned char *h_Bytestream;
	unsigned bytestreamSize;
	unsigned headerSize;

	unsigned height;
	unsigned width;
	unsigned bitsPerChannel;

	fastJpegExifSection_t *exifSections;
	unsigned exifSectionsCount;

	fastJpegQuantState_t quantState;
	fastJpegHuffmanState_t huffmanState;
	fastJpegScanStruct_t scanMap;
	unsigned restartInterval;
} fastJfifInfo_t;

typedef struct {
	fastJpegMode_t jpegMode;
	fastJpegFormat_t jpegFmt;

	unsigned char *d_Bytestream;
	unsigned bytestreamSize;

	unsigned height;
	unsigned width;
	unsigned bitsPerChannel;

	fastJpegExifSection_t *exifSections;
	unsigned exifSectionsCount;

	fastJpegQuantState_t quantState;
	fastJpegHuffmanState_t huffmanState;
	fastJpegScanStruct_t scanMap;
	unsigned restartInterval;
} fastJfifInfoAsync_t;

typedef struct {
	double sigma;
} fastGaussianFilter_t;

typedef struct {
	int columnOffset;
} fastColumnFilter_t;

typedef struct {
	unsigned char lut[256];
} fastLut_8_t;

typedef struct {
	unsigned char lut_R[256];
	unsigned char lut_G[256];
	unsigned char lut_B[256];
} fastLut_8_C_t;

typedef struct {
	unsigned short lut[256];
} fastLut_8_16_t;

typedef struct {
	unsigned short lut_R[256];
	unsigned short lut_G[256];
	unsigned short lut_B[256];
} fastLut_8_16_C_t;

typedef struct {
	unsigned short lut_R[256];
	unsigned short lut_G[256];
	unsigned short lut_B[256];

	fastBayerPattern_t pattern;
} fastLut_8_16_Bayer_t;

typedef struct {
	unsigned short lut[1024];
} fastLut_10_t;

typedef struct {
	unsigned short lut_R[1024];
	unsigned short lut_G[1024];
	unsigned short lut_B[1024];

	fastBayerPattern_t pattern;
} fastLut_10_16_Bayer_t;

typedef struct {
	unsigned char lut[4096];
} fastLut_12_8_t;

typedef struct {
	unsigned char lut_R[4096];
	unsigned char lut_G[4096];
	unsigned char lut_B[4096];
} fastLut_12_8_C_t;

typedef struct {
	unsigned short lut_R[4096];
	unsigned short lut_G[4096];
	unsigned short lut_B[4096];

	fastBayerPattern_t pattern;
} fastLut_12_16_Bayer_t;

typedef struct {
	unsigned short lut[4096];
} fastLut_12_t;

typedef struct {
	unsigned short lut_R[4096];
	unsigned short lut_G[4096];
	unsigned short lut_B[4096];
} fastLut_12_C_t;

typedef struct {
	unsigned short lut_R[16384];
	unsigned short lut_G[16384];
	unsigned short lut_B[16384];

	fastBayerPattern_t pattern;
} fastLut_14_16_Bayer_t;

typedef struct {
	unsigned short lut[16384];
} fastLut_16_t;

typedef struct {
	unsigned short lut_R[16384];
	unsigned short lut_G[16384];
	unsigned short lut_B[16384];
} fastLut_16_C_t;

typedef struct {
	unsigned short lut_R[16384];
	unsigned short lut_G[16384];
	unsigned short lut_B[16384];

	fastBayerPattern_t pattern;
} fastLut_16_Bayer_t;

typedef struct {
	unsigned short lut[65536];
} fastLut_16_FR_t;

typedef struct {
	unsigned short lut_R[65536];
	unsigned short lut_G[65536];
	unsigned short lut_B[65536];
} fastLut_16_FR_C_t;

typedef struct {
	unsigned short lut_R[65536];
	unsigned short lut_G[65536];
	unsigned short lut_B[65536];

	fastBayerPattern_t pattern;
} fastLut_16_FR_Bayer_t;

typedef struct {
	unsigned char lut[16384];
} fastLut_16_8_t;

typedef struct {
	unsigned char lut_R[16384];
	unsigned char lut_G[16384];
	unsigned char lut_B[16384];
} fastLut_16_8_C_t;

typedef struct {
	float *lutR;
	float *lutG;
	float *lutB;

	unsigned size1D;
} fastRGBLut_3D_t;

typedef struct {
	unsigned short lut12[4096];
	unsigned short lut14[16384];
} fastHdrGray_3x12_t;

typedef struct {
	int logOffset;
} fastHdrImport_t;

typedef struct {
	unsigned int dimH;
	unsigned int dimS;
	unsigned int dimV;

	fastColorSaturationOperationType_t operationH;
	fastColorSaturationOperationType_t operationS;
	fastColorSaturationOperationType_t operationV;

	float *LutH;
	float *LutS;
	float *LutV;
} fastHsvLut3D_t;

typedef struct {
	char *blackShiftMatrix;
	float *correctionMatrix;
} fastSam_t;

typedef struct {
	short *blackShiftMatrix;
	float *correctionMatrix;
} fastSam16_t;

typedef struct {
	fastBinningMode_t mode;
	unsigned factorX;
	unsigned factorY;
} fastBinning_t;

typedef struct {
	float matrix[12];
	int whiteLevel[3];
} fastBaseColorCorrection_t;

typedef struct {
	float Lut[3][1024];

	fastColorSaturationOperationType_t operation[3];
	fastColorSaturationChannelType_t sourceChannel[3];
} fastColorSaturation_t;

typedef struct {
	unsigned short toneCurve[1024];
} fastToneCurve_t;

typedef struct {
	fastBayerPattern_t pattern;
} fastBadPixelCorrection_t;

typedef struct {
	unsigned windowSize;
	unsigned short tint[3]; /*RGB*/
	float fi_tint;

	float fi_max;
	float coefficient;
} fastDefringe_t;

typedef struct {
	fastConvertType_t convert;
} fastExportParameters_t;

typedef struct {
	float R;
	float G1;
	float G2;
	float B;

	fastBayerPattern_t bayerPattern;
} fastWhiteBalance_t;

typedef struct {
	unsigned short divider;
	unsigned short* correctionMatrix;
} fastFFC_t;

typedef struct {
	float R;
	float G;
	float B;

	fastBayerPattern_t bayerPattern;
} fastBayerBlackShift_t;

typedef struct {
	unsigned overrideSourceBitsPerChannel;
	bool isOverrideSourceBitsPerChannel;
	unsigned targetBitsPerChannel;
} fastBitDepthConverter_t;

typedef struct {
	fastChannelType_t channel;
} fastSelectChannel_t;

typedef struct {
	float coefficientR;
	float coefficientG;
	float coefficientB;
} fastRgbToGrayscale_t;

typedef struct {
	unsigned char *data;
	unsigned width;
	unsigned pitch;
	unsigned height;
} fastChannelDescription_t;

typedef struct {
	fastBayerPattern_t bayerPattern;
} fastBayerPatternParam_t;

typedef struct {
	unsigned int stride;
} fastHistogramParade_t;

typedef struct {
	fastRGBAAlphaPadding_t padding;
} fastSDIRGBAExport_t;

typedef struct {
	unsigned overrideSourceBitsPerChannel;
} fastSDIYCbCrExport_t;

typedef struct {
	bool isConvert12to16;
} fastSDIRaw12Import_t;

typedef struct {
	fastDenoiseThresholdFunctionType_t function;
	fastWaveletType_t wavelet;
} fastDenoiseStaticParameters_t;

typedef struct {
	int dwt_levels;
	float enhance[3];
	float threshold[3];
	float threshold_per_level[33];
} fastDenoiseParameters_t;

typedef struct {
	unsigned short sdkLicenseVersion;
	unsigned short dongleLicenseVersion;
	char dongleName[56];
	char dongleId[8];
} fastLicenseProviderSenselockInfo_t;

typedef struct {
	int id;
} fastLicenseProviderGuardantFeature_t;

typedef struct {
	int id;
	int featuresCount;
	fastLicenseProviderGuardantFeature_t features[4];
} fastLicenseProviderGuardantProduct_t;

typedef struct {
	unsigned int dongleId;
	unsigned int productsCount;
	fastLicenseProviderGuardantProduct_t products[4];
} fastLicenseProviderGuardantInfo_t;

typedef struct {
	fastLicenseType_t licenseType;
	unsigned char sdkVersion[4];
	char buildDate[11];
	int remainingTrialDays;
	fastLicenseProvider_t licenseProvider;
	union {
		fastLicenseProviderSenselockInfo_t senselockInfo;
		fastLicenseProviderGuardantInfo_t guardantInfo;
	};
} fastLicenseInfo_t;

typedef struct {
	unsigned surfaceFmt;

	unsigned width;
	unsigned height;
	unsigned pitch;

	unsigned maxWidth;
	unsigned maxHeight;
	unsigned maxPitch;
} fastDeviceSurfaceBufferInfo_t;

typedef struct {
	char name[256];
	int major;
	int minor;
	int integrated;
	int isMultiGpuBoard;
	int	pciDeviceID;
	size_t totalMem;
} fastDeviceProperty;


typedef struct {
	fastComponentType_t componentType;
	void* param;
	size_t paramSize;
} fastPipelineParam_t;

/*---------------------Jpeg Encoder/Decoder----------------------------------*/
typedef struct {
	unsigned quality;
	bool async;
	bool isCustomeQuantTable;
	fastJpegQuantState_t* quantTable;
	fastJfifInfo_t* jfifInfo;
	fastJfifInfoAsync_t* jfifInfoAsync;
} fastJpegEncodeParam_t;

typedef struct {
	fastSurfaceFormat_t surfaceFmt;
	bool checkBytestream;
} fastJpegDecoderCreateParam_t;

/*---------------------Debayer----------------------------------*/
typedef struct {
	fastDebayerType_t debayerType;
} fastDebayerCreateParam_t;

typedef struct {
	fastBayerPattern_t bayerFmt;
} fastDebayerTransformParam_t;

/*---------------------Resize----------------------------------*/
typedef struct {
	unsigned maxDstWidth;
	unsigned maxDstHeight;

	double maxScaleFactor;

	float shiftX;
	float shiftY;
} fastResizerCreateParam_t;

typedef struct {
	fastResizeType_t resizeType;
	fastResizeMethod_t resizeMethod;
	unsigned resizedWidth;
	unsigned resizedHeight;
	fastRgb_t background;
} fastResizerTransformParam_t;

/*--------------------- Image Filter----------------------------------*/
typedef struct {
	fastImageFilterType_t filterType;
	void* staticFilterParameters;
} fastImageFilterCreate_t;

/*--------------------- Crop----------------------------------*/
typedef struct {
	unsigned maxDstWidth;
	unsigned maxDstHeight;
} fastCropCreateParam_t;

typedef struct {
	unsigned leftTopCoordsX;
	unsigned leftTopCoordsY;
	unsigned croppedWidth;
	unsigned croppedHeight;
} fastCropTransformParam_t;

/*--------------------- RawImport----------------------------------*/
typedef struct {
	fastRawFormat_t	rawFmt;
	void* staticParameters;
} fastRawImportCreateParam_t;

typedef struct {
	unsigned pitch;
} fastRawImportTransformParam_t;

/*--------------------- ImportFrom----------------------------------*/
typedef struct {
	fastSurfaceFormat_t surfaceFmt;
} fastImportFromCreateParam_t;

typedef struct {
	unsigned pitch;
} fastImportFromCopyParam_t;

/*--------------------- Export To----------------------------------*/
typedef struct {
	fastSurfaceFormat_t surfaceFmt;
} fastExportToCreateParam_t;

typedef struct {
	unsigned pitch;
} fastExportToCopyParam_t;

/*--------------------- Affine----------------------------------*/
typedef struct {
	unsigned char affineTransformMask;
} fastAffineCreateParam_t;

typedef struct {
	fastAffineTransformations_t affineType;
} fastAffineTransformParam_t;

/*--------------------- SDI Import From----------------------------*/
typedef struct {
	fastSDIFormat_t	sdiFmt;
	void* staticParameters;
} fastSDIImportFromCreateParam_t;

/*--------------------- SDI Export To----------------------------*/
typedef struct {
	fastSDIFormat_t	sdiFmt;
	void* staticParameters;
} fastSDIExportToCreateParam_t;

/*--------------------- SurfaceConverter-----------------------*/
typedef struct {
	fastSurfaceConverter_t surfaceConverterType;
	void* staticSurfaceConverterParameters;
} fastSurfaceConverterCreateParam_t;

/*--------------------- Denoise-----------------------*/
typedef struct {
	fastSurfaceFormat_t surfaceFmt;
	void* staticDenoiseParameters;
} fastDenoiseCreateParam_t;

#ifdef __cplusplus
}
#endif

#endif
