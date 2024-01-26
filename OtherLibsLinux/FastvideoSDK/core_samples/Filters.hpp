#pragma once

void cfaMedian(
    unsigned short* src,
    const unsigned srcPitch, // in Elements

    unsigned short* dst,
    const unsigned dstPitch, // in Elements

    const unsigned width,
    const unsigned height
);

void cfaBoxBlur(
    unsigned short* src,
    const unsigned srcPitch, // in Elements

    unsigned short* dst,
    const unsigned dstPitch, // in Elements

    const unsigned width,
    const unsigned height
);