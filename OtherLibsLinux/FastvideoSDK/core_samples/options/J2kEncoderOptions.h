
#ifndef __J2K_ENCODER_OPTIONS__
#define __J2K_ENCODER_OPTIONS__

#include "BaseOptions.h"

#include "fastvideo_encoder_j2k.h"

class J2kEncoderOptions : public BaseOptions {
public:
	char *AlgorithmName;
	fastEncoderJ2kAlgorithmType_t Algorithm;
    long InputFilesize;
    long TargetFilesize;
    float CompressionRatio;
    float Quality;
    int BatchSize;
    int CodeblockWidth;
    int CodeblockHeight;
	int DWT_Levels;

	int TileThreads; 
	int Tier2Threads;

	bool Discard;
	bool NoHeader;
    bool NoMCT;
	int TileWidth;
	int TileHeight;
	int OutputBitDepth;
	int OverwriteSurfaceBitDepth;
	int ss1_x, ss1_y, ss2_x, ss2_y, ss3_x, ss3_y;

    size_t MaxMemoryAvailable;
    size_t ReserveMemorySize;
    bool EnableMemoryReallocation;
    bool SequentialTiles;

	char* Comments;

    float Timeout;

	J2kEncoderOptions(void) {};
	~J2kEncoderOptions(void) {};

	virtual bool Parse(int argc, char *argv[]);
};

#endif // __J2K_ENCODER_OPTIONS__
