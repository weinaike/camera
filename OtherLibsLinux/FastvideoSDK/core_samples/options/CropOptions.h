
#ifndef __CROP_OPTIONS__
#define __CROP_OPTIONS__

class CropOptions {
private:

public:
	unsigned CropLeftTopCoordsX;
	unsigned CropLeftTopCoordsY;
	unsigned CropWidth;
	unsigned CropHeight;

	bool IsEnabled;

	const char * OptionName;

	CropOptions(void) : CropLeftTopCoordsX(0), CropLeftTopCoordsY(0), CropWidth(0), CropHeight(0), IsEnabled(false) {
		OptionName = "crop";
	}
	CropOptions(const char *optionName) : CropLeftTopCoordsX(0), CropLeftTopCoordsY(0), CropWidth(0), CropHeight(0),
		IsEnabled(false) {
		OptionName = optionName;
	}
	~CropOptions(void) { }

	bool Parse(int argc, char *argv[]);
};

#endif // __CROP_OPTIONS__
