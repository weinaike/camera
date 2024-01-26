#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>

#include "FfcGenerator.hpp"
#include "HelperFFC.hpp"
#include "Filters.hpp"
#include "SurfaceTraits.hpp"
#include "EnumToStringSdk.h"

int Filtering(
    unsigned short* srcImage,
    const unsigned srcPitch, // in Element size

    const unsigned width,
    const unsigned height,

    unsigned short* ffc,
    const unsigned ffcPitch // in Element size
) {
    std::unique_ptr<unsigned short> cfaTmp;
    {
        cfaTmp.reset(new unsigned short[srcPitch * height]);
        memset(cfaTmp.get(), 0, srcPitch * height * sizeof(unsigned short));
    }

    cfaMedian(srcImage, srcPitch, cfaTmp.get(), srcPitch, width, height);

    // restore line 0 & column 0
    {
        unsigned short* ptr = cfaTmp.get();
        for (unsigned y = 0; y < height; y++) {
            ptr[y * srcPitch] = ptr[y * srcPitch + 1];
        }

        memcpy(ptr, &ptr[srcPitch], srcPitch * sizeof(unsigned short));
    }

    cfaBoxBlur(cfaTmp.get(), srcPitch, ffc, ffcPitch, width, height);

    return 0;
}

unsigned short GetDivider(
    unsigned short* src,
    const unsigned srcPitch,

    const unsigned width,
    const unsigned height
) {
    const unsigned y = 2 * (height >> 2);
    const unsigned x = 2 * (width >> 2);
    return std::max<unsigned short>(0, src[y * srcPitch + x]);
}

float GetExtrapolateCoeff(
    const unsigned value
) {
    const int rem = value & 3;
    if (rem == 0) {
        return 4.f / 3.f;
    } else if (rem == 2) {
        return 4.f;
    } else if (rem == 3) {
        return 2.f;
    }
    return 0.f;
}

void ExtrapolateBorder(
    unsigned short* src,
    const unsigned srcPitch,

    unsigned short* dst,
    const unsigned dstPitch,

    const uint2 srcSize,
    const uint2 dstSize
) {
    float multiply = GetExtrapolateCoeff(dstSize.x);

    // extrapolate last element in line
    for (unsigned ySrc = 0, yDst = 0; ySrc < srcSize.y; ySrc += 4, yDst++) {
        const int delta = (int)src[ySrc * srcPitch + srcSize.x - 1] - (int)dst[yDst * dstPitch + dstSize.x - 2];
        int coeff = std::floor(delta * multiply + 0.5f);

        dst[yDst * dstPitch + dstSize.x - 1] = (int)dst[yDst * dstPitch + dstSize.x - 2] + coeff;
    }

    multiply = GetExtrapolateCoeff(dstSize.y);

    // extrapolate element in last line
    for (unsigned xSrc = 0, xDst = 0; xSrc < srcSize.x; xSrc += 4, xDst++) {
        const int delta = (int)src[(srcSize.y - 1) * srcPitch + xSrc] - (int)dst[(dstSize.y - 2) * dstPitch + xDst];
        int coeff = std::floor(delta * multiply + 0.5f);

        dst[(dstSize.y - 1) * dstPitch + xDst] = (int)dst[(dstSize.y - 2) * dstPitch + xDst] + coeff;
    }

    // extrapolate bottom right element
    {
        const int delta = (int)dst[(dstSize.y - 2) * dstPitch + dstSize.x - 1] - (int)dst[(dstSize.y - 3) * dstPitch + dstSize.x - 1];
        int coeff = std::floor(delta * multiply + 0.5f);

        dst[(dstSize.y - 1) * dstPitch + dstSize.x - 1] = (int)dst[(dstSize.y - 2) * dstPitch + dstSize.x - 1] + coeff;
    }
}

unsigned CompactFFC(
    bool disableFiltering,
    Image<FastAllocator>& inputImg,
    Image<FastAllocator>& compactedFfc
) {
    // normalize values to I16
    if (inputImg.surfaceFmt != FAST_I16) {
        const int diff = 16 - GetBitsPerChannelFromSurface(inputImg.surfaceFmt);

        unsigned short* ptr = (unsigned short*)inputImg.data.get();
        unsigned short srcPitch = inputImg.wPitch / sizeof(unsigned short);

        inputImg.surfaceFmt = FAST_I16;
        for (unsigned y = 0; y < inputImg.h; y++) {
            for (unsigned x = 0; x < inputImg.w; x++) {
                ptr[y * srcPitch + x] <<= diff;
            }
        }
    }

    // get FFC coeffs
    const unsigned ffcWidth = inputImg.w; // source FFC the same as input image
    const unsigned ffcHeight = inputImg.h;
    const unsigned ffcPitch = inputImg.wPitch / sizeof(unsigned short); // in ushort size
    std::unique_ptr<unsigned short> ffc;
    {
        ffc.reset(new unsigned short[ffcPitch * ffcHeight]);
        memset(ffc.get(), 0, ffcPitch * ffcHeight * sizeof(unsigned short));
    }

    if (!disableFiltering) {
        Filtering(
            (unsigned short*)inputImg.data.get(),
            inputImg.wPitch / sizeof(unsigned short), // in Element size

            inputImg.w, inputImg.h,

            ffc.get(), ffcPitch
        );
    } else {
        memcpy((unsigned char*)ffc.get(), inputImg.data.get(), inputImg.wPitch * inputImg.h);
    }

    // compacting FFC matrix
    {
        compactedFfc.surfaceFmt = FAST_I16;
        compactedFfc.w = GetCompactedFfcWidth(inputImg.w);
        compactedFfc.wPitch = GetPitchFromSurface(compactedFfc.surfaceFmt, compactedFfc.w);
        compactedFfc.h = GetCompactedFfcHeight(inputImg.h);
        compactedFfc.Allocate();

        unsigned short* src = ffc.get();
        unsigned short* dst = (unsigned short*)compactedFfc.data.get();
        const unsigned dstPitch = compactedFfc.wPitch / sizeof(unsigned short);
        for (unsigned ySrc = 0, yDst = 0; ySrc < ffcHeight; ySrc += 4, yDst++) {
            for (unsigned xSrc = 0, xDst = 0; xSrc < ffcWidth; xSrc += 4, xDst++) {
                dst[yDst * dstPitch + xDst] = src[ySrc * ffcPitch + xSrc];
            }
        }

        ExtrapolateBorder(
            src, ffcPitch,
            dst, dstPitch,
            make_uint2(ffcWidth, ffcHeight),
            make_uint2(compactedFfc.w, compactedFfc.h)
        );
    }

    return GetDivider(
        (unsigned short*)compactedFfc.data.get(), compactedFfc.wPitch / sizeof(unsigned short),
        compactedFfc.w, compactedFfc.h
    );
}
