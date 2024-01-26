#ifndef __METRICS_HPP__
#define __METRICS_HPP__

#include <vector>

class Metrics {
private:
	unsigned char *etalonImg;
	unsigned char *testImg;
	unsigned width;
	unsigned height;
	unsigned channels;

	void receiveColor(double *out1, double *out2, int colorOffset);
	
public:
	struct MSEResult {
		double R;
		double G;
		double B;
	};

	Metrics(unsigned char *etalonImg, unsigned char *testImg, unsigned width, unsigned height, unsigned channels) :
			etalonImg(etalonImg), testImg(testImg), width(width), height(height), channels(channels)
	{ 	};

	MSEResult MseByComponents (int skipBorderWidth);
	double MSE (int skipBorderWidth);
	double PSNR (int skipBorderWidth);
	double SSIM(unsigned long offset, unsigned long interleave);
	void DiffGistogram (std::vector<int> &res, int skipBorderWidth);
	void GetPixelDiff(unsigned char *diff, int skipBorderWidth);
};


#endif
