
#ifndef __CUBE_LUT__
#define __CUBE_LUT__

#include <vector>
#include <string>

using namespace std;

class CubeLUT {
public:
	typedef vector<float> tableRow;
	typedef vector<tableRow> table1D;
	typedef vector<table1D> table2D;
	typedef vector <table2D> table3D;
	enum LUTState {
		OK = 0,
		NotInitialized = 1,
		ReadError = 10,
		WriteError,
		PrematureEndOfFile,
		LineError,
		UnknownOrRepeatedKeyword = 20,
		TitleMissingQuote,
		DomainBoundsReversed,
		LUTSizeOutOfRange,
		CouldNotParseTableData
	};
	LUTState status;
	string title;
	tableRow domainMin;
	tableRow domainMax;
	table1D LUT1D;
	table3D LUT3D;
	CubeLUT(void) {
		status = NotInitialized;
	};
	LUTState LoadCubeFile(ifstream &infile);
	LUTState SaveCubeFile(ofstream &outfile);
private:
	string ReadLine(ifstream &infile, char lineSeparator);
	tableRow ParseTableRow(const string &lineOfText);
};

#endif //__CUBE_LUT__
