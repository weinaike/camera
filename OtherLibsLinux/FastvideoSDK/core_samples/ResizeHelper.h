

#pragma once

const double resizeCorrectionCoefficient = 1.01;

inline double GetResizeMaxScaleFactor(unsigned maxWidth, unsigned maxHeight, ResizerOptions options)
{
	double maxScaleFactorX = double(maxWidth) / double(options.OutputWidth);
	double maxScaleFactor = maxScaleFactorX;
	if (options.OutputHeightEnabled) {
		double maxScaleFactorY = double(maxHeight) / double(options.OutputHeight);
		maxScaleFactor = maxScaleFactorX > maxScaleFactorY ? maxScaleFactorX : maxScaleFactorY;
	}
	maxScaleFactor *= resizeCorrectionCoefficient;
	return maxScaleFactor;
}

inline unsigned GetResizeMaxHeight(unsigned maxHeight, double maxScaleFactor)
{
	return static_cast<unsigned>(resizeCorrectionCoefficient * resizeCorrectionCoefficient * maxHeight / maxScaleFactor);
}
