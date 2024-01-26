#ifndef __SSIM_H__
#define __SSIM_H__

float ssimFactor(const double *img1, const double *img2,
			unsigned long width, unsigned long height,
			double scale, bool doluminance,
			unsigned long offset, unsigned long interleave);

#endif
