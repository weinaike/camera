cmake_minimum_required(VERSION 3.18)

OPTION(USE_CUDA  "使能CUDA库"   ON)
message(STATUS "USE_CUDA: ${USE_CUDA}")

if(USE_CUDA)
    if(NOT WIN32)
        set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
    endif()

    project(gpu-camera-sample VERSION 1.0 LANGUAGES CUDA CXX)
    # 设置宏
    add_definitions(-DUSE_CUDA)
    # add_definitions(-DENABLE_GL)
    set(CMAKE_CXX_STANDARD 11)
else()
    project(gpu-camera-sample VERSION 1.0 LANGUAGES CXX)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)



if(WIN32)
    set(ARCHITECTURE WIN32)
#    find_library(UCRTBASED_LIB ucrtbased PATHS "C:/Program Files (x86)/Windows Kits/10/bin/x64/ucrt/")
#    if(UCRTBASED_LIB)
#        message("Found ucrtbased.dll: ${UCRTBASED_LIB}")
#    else()
#        message(FATAL_ERROR "Could not find ucrtbased.dll")
#    endif()

else()
    EXECUTE_PROCESS( COMMAND uname -m COMMAND tr -d '\n' OUTPUT_VARIABLE ARCHITECTURE )
endif()

message("target architecture ${ARCHITECTURE}")

option(USE_XIMEA ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(Qt5 COMPONENTS Gui Widgets OpenGL Network Charts REQUIRED)
    # 打印查找到的CUDA库
    message(STATUS "CUDA libraries: ${CUDAToolkit_LIBRARY_DIR}")
else()
    find_package(Qt5 COMPONENTS Gui Widgets Network Charts REQUIRED)
    message(STATUS "No CUDA libraries")
endif()



set(ADDITIONAL_INCDIR)

if(WIN32)
    set(LIB_DIR OtherLibs)
    set(FASTLIB_DIR ${LIB_DIR}/fastvideoSDK)
    set(FastVideo_PATH ${FASTLIB_DIR}/fastvideo_sdk)
else()
    set(LIB_DIR OtherLibsLinux)
    set(FASTLIB_DIR ${LIB_DIR}/FastvideoSDK)
    set(FastVideo_PATH ${FASTLIB_DIR}/fastvideo_sdk)
endif()

if(USE_XIMEA)
    add_definitions(-DSUPPORT_XIMEA)
    set(Ximea_PATH D:/devs/XIMEA/API/xiAPI/)
    find_path(Ximea_INC
        NAMES xiApi.h
        PATHS ${Ximea_PATH}
        REQUIRED)
    find_library(Ximea_LIB
        NAMES xiapi64
        PATHS ${Ximea_PATH}
        REQUIRED)

    if(WIN32)
        set(THIRD_PARTY_DLLS ${THIRD_PARTY_DLLS} ${Ximea_PATH}/xiapi64.dll)
    endif()
    set(ADDITIONAL_INCDIR ${ADDITIONAL_INCDIR} ${Ximea_INC})
endif()

find_path(FastVideo_INC
    NAMES fastvideo_sdk_define.h
    PATHS ${FastVideo_PATH}/inc/
    REQUIRED)

set(FFMPEG_PATH ${LIB_DIR}/ffmpeg)
find_path(FFMPEG_INC
    NAMES libavutil/opt.h
    PATHS ${FFMPEG_PATH}/include
    REQUIRED)
find_library(AVCODEC_LIB
    NAMES avcodec
    PATHS ${FFMPEG_PATH}/lib
    REQUIRED)
find_library(AVUTIL_LIB
    NAMES avutil
    PATHS ${FFMPEG_PATH}/lib
    REQUIRED)
find_library(AVFORMAT_LIB
    NAMES avformat
    PATHS ${FFMPEG_PATH}/lib
    REQUIRED)
find_library(AVRESAMPLE_LIB
    NAMES swresample
    PATHS ${FFMPEG_PATH}/lib
    REQUIRED)

find_library(LIBSWSCALE_LIB
    NAMES swscale
    PATHS /usr/local/lib /usr/lib
    )    
set(FFMPEG_LIB ${AVCODEC_LIB} ${AVUTIL_LIB} ${AVFORMAT_LIB} ${AVRESAMPLE_LIB} ${LIBSWSCALE_LIB})

find_path(JPEG_INC
    NAMES jpeglib.h
    PATHS ${LIB_DIR}/jpeg-turbo/include/
    REQUIRED)

find_path(ZJV_INC
    NAMES PublicPipeline.h
    PATHS ${LIB_DIR}/zjv/inc/
    REQUIRED)

find_library(ZJV_LIB
    NAMES zj_pipe
    PATHS ${LIB_DIR}/zjv/libs
    REQUIRED)

find_library(ZJV_LIB_CUDA
    NAMES cuda_lib
    PATHS ${LIB_DIR}/zjv/libs
    REQUIRED)

find_library(ONNX
    NAMES onnxruntime
    PATHS ${LIB_DIR}/zjv/libs
    REQUIRED)


# set(TENSORRT_ROOT_DIR /usr/local/TensorRT-8.6.1.6_cuda11.8/)
set(TENSORRT_ROOT_DIR /usr/local/TensorRT-8.6.1.6_cuda12.0/)

include_directories(${TENSORRT_ROOT_DIR}/include)
link_directories(${TENSORRT_ROOT_DIR}/lib)

set(TRT_LIBS nvinfer nvonnxparser  ${TENSORRT_ROOT_DIR}/lib/libnvinfer_plugin.so.8)


# set(CUNDD_ROOT_DIR /usr/local/cudnn-cuda11/)
set(CUNDD_ROOT_DIR /usr/local/cudnn-cuda12/)

include_directories(${CUNDD_ROOT_DIR}/include)
link_directories(${CUNDD_ROOT_DIR}/lib)

set(CUDNN_LIBS cudnn)


if(WIN32)
    find_library(JPEG_LIB
        NAMES jpeg62_8bit
        PATHS ${LIB_DIR}/jpeg-turbo/lib/
        REQUIRED)
    # find_library(TURBOJPEG_LIB
    #     NAMES turbojpeg-static
    #     PATHS OtherLibs/jpeg-turbo/lib/
    #     REQUIRED)
    if(USE_CUDA)
        set(ADDITIONAL_LIBS ${CUDAToolkit_LIBRARY_DIR}/cuda.lib opengl32)
    endif()
else()
    set(JPEG_LIB jpeg)
    set(TURBOJPEG_LIB jpeg)
    if(USE_CUDA)
        set(ADDITIONAL_LIBS cuda GL pthread)
    else()
        set(ADDITIONAL_LIBS pthread)
    endif()
endif()


if(WIN32)
    
    set(THIRD_PARTY_DLLS ${THIRD_PARTY_DLLS}
        ${LIB_DIR}/ffmpeg/lib/avcodec-58.dll
        ${LIB_DIR}/ffmpeg/lib/avformat-58.dll
        ${LIB_DIR}/ffmpeg/lib/avutil-56.dll
        ${LIB_DIR}/ffmpeg/lib/swresample-3.dll
        ${LIB_DIR}/jpeg-turbo/lib/jpeg62_8bit.dll
        
            
       
    )
    if(USE_CUDA)
        set(THIRD_PARTY_DLLS ${THIRD_PARTY_DLLS}
            ${CUDAToolkit_LIBRARY_ROOT}/bin/cudart64_110.dll
        )
    endif()
endif()


set(ADDITIONAL_LIBS
    ${ADDITIONAL_LIBS}
    ${Ximea_LIB}
    ${ZJV_LIB}
    ${ZJV_LIB_CUDA}
    ${ONNX}
    ${TRT_LIBS}
    ${CUDNN_LIBS}
    ${FFMPEG_LIB}
)
if(USE_CUDA)
    if(NOT ${ARCHITECTURE} STREQUAL "aarch64")
        find_path(VideoCodec_INC
            NAMES cuviddec.h
            PATHS ${LIB_DIR}/nvcodecs/include/ ${LIB_DIR}/nvcodecs/Interface/
            REQUIRED)
        find_library(VideoCodec_LIB
            NAMES nvcuvid
            PATHS ${LIB_DIR}/nvcodecs/Lib/x64/ ${LIB_DIR}/nvcodecs/Lib/linux/stubs/x86_64
            REQUIRED)

        set(ADDITIONAL_INCDIR ${ADDITIONAL_INCDIR}
            ${VideoCodec_INC})

        set(ADDITIONAL_LIBS ${ADDITIONAL_LIBS}
            ${VideoCodec_LIB})
    endif()
endif()


#find_library(VideoCodec2_LIB
#    NAMES nvencodeapi
#    PATHS OtherLibs/nvcodecs/Lib/Win32/
#    REQUIRED)

set(ADDITIONAL_INCDIR ${ADDITIONAL_INCDIR}
    ${FastVideo_INC}
    ${FastVideo_INC}/../../common
    ${FastVideo_INC}/../../core_samples
    ${FFMPEG_INC}
    ${JPEG_INC}
    ${ZJV_INC}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(src/CameraSample)

# create a list of files to copy
set(THIRD_PARTY_DLLS ${THIRD_PARTY_DLLS}
#    ${FASTLIB_DIR}/fastvideo_sdk/bin/x64/fastvideo_sdk.dll
)
message(STATUS "THIRD_PARTY_DLLS: ${THIRD_PARTY_DLLS}")
install(FILES ${THIRD_PARTY_DLLS}
    DESTINATION ${CMAKE_BINARY_DIR}/bin/
)
