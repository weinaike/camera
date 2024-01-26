

#ifndef __J2K_DECODER_OPTIONS__
#define __J2K_DECODER_OPTIONS__

#include "BaseOptions.h"

class J2kDecoderOptions : public BaseOptions {
	void Init();
	
public:
    // Constraints
    unsigned ResolutionLevels;
    unsigned DecodePasses;
    size_t MaxMemoryAvailable; 
    size_t ReserveMemorySize;
	bool EnableMemoryReallocation;
	bool ForceTo8bits;
		
    // Benchmarks
	bool Discard;
    double Timeout;

    // Speedup
    unsigned MaxBatchSize;
    unsigned Tier2Threads;

    // Tiles
	unsigned MaxTileWidth;
    unsigned MaxTileHeight;
    bool SequentialTiles; 

    // Window
    unsigned WindowLeftTopCoordsX;
    unsigned WindowLeftTopCoordsY;
    unsigned WindowWidth;
    unsigned WindowHeight;
	bool IsEnabledWindow;

	bool PrintBoxes;
	bool PrintGML;
    char* OutputPathGML;

	J2kDecoderOptions();

	J2kDecoderOptions(bool ignoreOutput);

	virtual bool Parse(int argc, char *argv[]);
};
#endif // __J2K_DECODER_OPTIONS__
