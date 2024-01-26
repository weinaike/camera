#include <cstdio>
#include <algorithm>
#include <cmath>
#include <memory>

template<typename T>
unsigned short clip(T x, const int windowSize) {
    return (int)(((float)x + 0.5f) / windowSize);
}

void cfaMedian(
    unsigned short* src,
    const unsigned srcPitch, // in Elements

    unsigned short* dst,
    const unsigned dstPitch, // in Elements

    const unsigned width,
    const unsigned height
) {
    unsigned short area[9];
    for (unsigned y = 1; y < height - 1; y++) {
        for (unsigned x = 1; x < width - 1; x++) {
            //1st y
            area[0] = src[(y - 1) * srcPitch + x - 1];
            area[1] = src[(y - 1) * srcPitch + x    ];
            area[2] = src[(y - 1) * srcPitch + x + 1];

            //2nd y
            area[3] = src[(y    ) * srcPitch + x - 1];
            area[4] = src[(y    ) * srcPitch + x    ];
            area[5] = src[(y    ) * srcPitch + x + 1];

            //3rd y
            area[6] = src[(y + 1) * srcPitch + x - 1];
            area[7] = src[(y + 1) * srcPitch + x    ];
            area[8] = src[(y + 1) * srcPitch + x + 1];

            std::sort(area, area + 9);
            dst[y * dstPitch + x] = area[4];
        }
    }
}

void cfaBoxBlurX(
    unsigned short* src,
    const unsigned srcPitch, // in Elements

    int* cfatmp,

    const unsigned width,
    const unsigned height
) {
    const int mBoxW = 16;
    for (int row = 0; row < height; row++) {
        int tmp = 0;
        for (int col = 0; col <= mBoxW; col++) {
            tmp += (int)src[row * srcPitch + col];
        }
        cfatmp[row * width + 0] = tmp;

        for (int col = 1; col <= mBoxW; col++) {
            cfatmp[row * width + col] = cfatmp[row * width + col - 1] + src[row * srcPitch + mBoxW + col];
        }

        int len = mBoxW + 1;
        for (int col = mBoxW + 1; col < width - mBoxW; col++) {
            cfatmp[row * width + col] =
                cfatmp[row * width + col - 1] +
                src[row * srcPitch + mBoxW + col] -
                clip(cfatmp[row * width + col - mBoxW - 1], len);
            len = std::min(len + 1, 33);
        }

        for (int col = width - mBoxW; col < width; col++) {
            cfatmp[row * width + col] =
                cfatmp[row * width + col - 1] -
                clip(cfatmp[row * width + col - mBoxW - 1], len);
        }
    }

    for (unsigned y = 0; y < height; y++) {
        int len = mBoxW;
        for (unsigned x = 0; x <= mBoxW; x++) {
            cfatmp[y * width + x] = clip(cfatmp[y * width + x], len + 1);
            len++;
        }

        for (unsigned x = mBoxW + 1; x < width - mBoxW; x++) {
            cfatmp[y * width + x] = clip(cfatmp[y * width + x], len); // len = 33
        }

        for (unsigned x = width - mBoxW; x < width; x++) {
            len--;
            cfatmp[y * width + x] = clip(cfatmp[y * width + x], len);
        }
    }
}

void cfaBoxBlurY(
    int* cfatmp,

    unsigned short* dst,
    const unsigned dstPitch, // in Elements

    const unsigned width,
    const unsigned height
) {
    const int mBoxH = 16;

    std::unique_ptr<int> tmpBuffer(new int[width * height]);
    int* cfa2tmp = tmpBuffer.get();

    for (int col = 0; col < width; col++) {
        int tmp = 0;
        for (int row = 0; row <= mBoxH; row++) {
            tmp += cfatmp[row * width + col];
        }
        cfa2tmp[0 * width + col] = tmp;

        for (int row = 1; row <= mBoxH; row++) {
            cfa2tmp[row * width + col] = cfa2tmp[(row - 1) * width + col] + cfatmp[(row + mBoxH) * width + col];
        }

        int len = mBoxH + 1;
        for (int row = mBoxH + 1; row < height - mBoxH; row++) {
            cfa2tmp[row * width + col] =
                cfa2tmp[(row - 1) * width + col] +
                cfatmp[(row + mBoxH) * width + col] -
                clip(cfa2tmp[(row - mBoxH - 1) * width + col], len);
            len = std::min(len + 1, 33);
        }

        for (int row = height - mBoxH; row < height; row++) {
            cfa2tmp[row * width + col] =
                cfa2tmp[(row - 1) * width + col] -
                clip(cfa2tmp[(row - mBoxH - 1) * width + col], len);
        }
    }

    for (unsigned x = 0; x < width; x++) {
        int len = mBoxH;
        for (unsigned y = 0; y <= mBoxH; y++) {
            dst[y * dstPitch + x] = clip(cfa2tmp[y * width + x], len + 1);
            len++;
        }

        for (unsigned y = mBoxH + 1; y < height - mBoxH; y++) {
            dst[y * dstPitch + x] = clip(cfa2tmp[y * width + x], len); // len = 33
        }

        for (unsigned y = height - mBoxH; y < height; y++) {
            len--;
            dst[y * dstPitch + x] = clip(cfa2tmp[y * width + x], len);
        }
    }
}

void cfaBoxBlur(
    unsigned short* src,
    const unsigned srcPitch, // in Elements

    unsigned short* dst,
    const unsigned dstPitch, // in Elements

    const unsigned width,
    const unsigned height
) {
    std::unique_ptr<int> tmpBuffer(new int[width * height]);

    cfaBoxBlurX(
        src, srcPitch,
        tmpBuffer.get(),
        width, height
    );

    cfaBoxBlurY(
        tmpBuffer.get(),
        dst, dstPitch,
        width, height
    );
}
