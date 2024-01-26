#ifndef HELPER_COMMON_H
#define HELPER_COMMON_H

#include <algorithm>
#include <assert.h>

// namespace unnamed (internal)
namespace
{
    //! size of PGM file header
    const unsigned int PGMHeaderSize = 0x40;

    // types

    //! Data converter from unsigned char / unsigned byte to type T
    template<class T>
    struct ConverterFromUByte;

    //! Data converter from unsigned char / unsigned byte
    template<>
    struct ConverterFromUByte<unsigned char>
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        float operator()(const unsigned char &val)
        {
            return static_cast<unsigned char>(val);
        }
    };

    //! Data converter from unsigned char / unsigned byte to float
    template<>
    struct ConverterFromUByte<float>
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        float operator()(const unsigned char &val)
        {
            return static_cast<float>(val);
        }
    };

    //! Data converter from unsigned char / unsigned byte to type T
    template<class T>
    struct ConverterToUByte;

    //! Data converter from unsigned char / unsigned byte to unsigned int
    template<>
    struct ConverterToUByte<unsigned char>
    {
        //! Conversion operator (essentially a passthru
        //! @return converted value
        //! @param  val  value to convert
        unsigned char operator()(const unsigned char &val)
        {
            return val;
        }
    };

    //! Data converter from unsigned char / unsigned byte to unsigned int
    template<>
    struct ConverterToUByte<float>
    {
        //! Conversion operator
        //! @return converted value
        //! @param  val  value to convert
        unsigned char operator()(const float &val)
        {
		    return static_cast<unsigned char>(val < 0.0f ? 0 : ( val > 255.0f ? 255 : val ));
        }
    };
}

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result != 0)
#endif
#ifndef SSCANF
#define SSCANF sscanf_s
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#ifndef FOPEN_FAIL
#define FOPEN_FAIL(result) (result == NULL)
#endif
#ifndef SSCANF
#define SSCANF sscanf
#endif
#endif

template<typename T> static inline T _uSnapUp(T a, T b){
    return a + (b - a % b) % b;
}

#endif
