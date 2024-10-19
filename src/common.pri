#QMAKE_CXXFLAGS += "/openmp"

QMAKE_CXXFLAGS_RELEASE -= -Zc:strictStrings
QMAKE_CFLAGS_RELEASE -= -Zc:strictStrings
QMAKE_CFLAGS -= -Zc:strictStrings
QMAKE_CXXFLAGS -= -Zc:strictStrings
QMAKE_CXXFLAGS += -execution-charset:utf-8

QMAKE_LFLAGS_WINDOWS += "/MACHINE:X64"
QMAKE_LFLAGS_WINDOWS += "/LARGEADDRESSAWARE"
QMAKE_LFLAGS_WINDOWS += "/ENTRY:mainCRTStartup"

OTHER_LIB_PATH = $$dirname(PWD)/OtherLibs

JPEGTURBO = $$OTHER_LIB_PATH/jpeg-turbo

ONNX = $$OTHER_LIB_PATH/onnx/win_x64/OnnxRuntime
ONNX_INC = $$ONNX/include
ONNX_LIB = $$ONNX/lib

SNAP7 = $$OTHER_LIB_PATH/snap7/
#SNAP7_INC = $$SNAP7/c-cpp
SNAP7_LIB = $$SNAP7/win64

ZJV = $$OTHER_LIB_PATH/zjv
ZJV_INC = $$ZJV/inc
ZJV_INC += $$ZJV/inc/public


CONFIG(debug, debug|release){
    ZJV_LIB = $$ZJV/lib/Debug
}else:CONFIG(release, debug|release){
    ZJV_LIB = $$ZJV/lib/Release
}

FASTVIDEOPATH = $$OTHER_LIB_PATH/FastvideoSDK
FASTVIDEO_SDK = $$FASTVIDEOPATH/fastvideo_sdk

FASTVIDEO_INC  = $$FASTVIDEO_SDK/inc
FASTVIDEO_INC += $$FASTVIDEOPATH/common
#FASTVIDEO_INC += $$FASTVIDEOPATH/libs/OpenGL/inc

FFMPEG_PATH = $$OTHER_LIB_PATH/ffmpeg
#FFMPEG_SRC  = $$FFMPEG_PATH/src/ffmpeg-3.4.2.tar.gz
FFMPEG_LIB  = $$FFMPEG_PATH/lib

FLIR_PATH = $$OTHER_LIB_PATH/FLIR
IMPERX_PATH = $$OTHER_LIB_PATH/Imperx

#FASTVIDEO_LIB  = -L$$FASTVIDEO_SDK/lib/$$PLATFORM
#FASTVIDEO_DLL_PATH = $$FASTVIDEO_SDK/bin/$$PLATFORM

#FASTVIDEO_DLL  = $$FASTVIDEO_DLL_PATH/fastvideo_sdk.dll
#FASTVIDEO_DLL += $$FASTVIDEO_DLL_PATH/fastvideo_mjpeg.dll

#FASTVIDEO_DLL += $$FASTVIDEO_DLL_PATH/fastvideo_experimentalImageFilter.dll
#FASTVIDEO_DLL += $$FASTVIDEO_DLL_PATH/fastvideo_denoise.dll

#FASTVIDEO_EXTRA_DLLS += $$FASTVIDEOPATH/libs/senselock/$$PLATFORM/sense4.dll

#FASTVIDEO_DLL += $$FASTVIDEO_DLL_PATH/fastvideo_nppFilter.dll
#FASTVIDEO_DLL += $$FASTVIDEO_DLL_PATH/fastvideo_nppResize.dll
#FASTVIDEO_DLL += $$FASTVIDEO_DLL_PATH/fastvideo_nppGeometry.dll

#FASTVIDEO_LIB += -lfastvideo_sdk \
#        -lfastvideo_mjpeg \
#        -lfastvideo_denoise
#        -lfastvideo_experimentalImageFilter \
#        -lfastvideo_nppFilter \
#        -lfastvideo_nppResize \
#        -lfastvideo_jpegLosslessDecoder \
#        -lfastvideo_nppGeometry

#FASTVIDEO_EXTRA_LIBS += -L$$FASTVIDEOPATH/libs/OpenGL/lib/win/$$PLATFORM
FASTVIDEO_EXTRA_DLLS += $$FFMPEG_PATH/lib/avcodec-58.dll
FASTVIDEO_EXTRA_DLLS += $$FFMPEG_PATH/lib/avformat-58.dll
FASTVIDEO_EXTRA_DLLS += $$FFMPEG_PATH/lib/avutil-56.dll
FASTVIDEO_EXTRA_DLLS += $$FFMPEG_PATH/lib/swresample-3.dll


FASTVIDEO_EXTRA_DLLS +=  $$JPEGTURBO/lib/jpeg62_8bit.dll

FASTVIDEO_EXTRA_DLLS +=  $$ZJV_LIB/zj_pipe.dll
FASTVIDEO_EXTRA_DLLS +=  $$ONNX_LIB/onnxruntime.dll
FASTVIDEO_EXTRA_DLLS +=  $$SNAP7_LIB/snap7.dll


#FASTVIDEO_EXTRA_DLLS += $$FFMPEG_SRC

#FASTVIDEO_EXTRA_DLLS += $$[QT_INSTALL_BINS]/icudt58.dll
#FASTVIDEO_EXTRA_DLLS += $$[QT_INSTALL_BINS]/icuin58.dll
#FASTVIDEO_EXTRA_DLLS += $$[QT_INSTALL_BINS]/icuuc58.dll
#FASTVIDEO_EXTRA_DLLS += $$LIBTIFF_LIB/tiff.dll

CUDA_TOOLKIT_PATH = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0"
#CUDA_TOOLKIT_PATH = "/usr/local/cuda"
CUDAINC += $$CUDA_TOOLKIT_PATH/include
CUDA_DLL_PATH = $$CUDA_TOOLKIT_PATH/bin
CUDA_DLL += $$CUDA_DLL_PATH/cudart64_12.dll
CUDA_DLL += $$CUDA_DLL_PATH/cublas64_12.dll

#CUDA_DLL += $$CUDA_DLL_PATH/nppc64_12.dll
#CUDA_DLL += $$CUDA_DLL_PATH/nppif64_12.dll
#CUDA_DLL += $$CUDA_DLL_PATH/nppicc64_12.dll
#CUDA_DLL += $$CUDA_DLL_PATH/nppig64_12.dll

#CUDA_LIB += -lnppicc
#CUDA_LIB += -lnppig
#CUDA_LIB += -lnppif
#CUDA_LIB += -lnpps
#CUDA_LIB += -lnppc
#CUDA_LIB += -lcudart
#CUDA_LIB += -lcuda
#CUDA_LIB += -lcublas
#CUDA_LIB += -lnvml

# NVIDIA VIDEO CODEC SDK
# https://developer.nvidia.com/nvidia-video-codec-sdk/download
#NVCODECS = $$OTHER_LIB_PATH/nvcodecs
#INCLUDEPATH += $$NVCODECS/include
#LIBS += -L$$NVCODECS/Lib/$$PLATFORM -lnvcuvid

# TensorRT

TRT_PATH = "D:/code/TensorRT-8.6.1.6"
TRTINC += $$TRT_PATH/include
TRT_DLL_PATH = $$TRT_PATH/lib
TRT_DLLS += $$TRT_PATH/nvinfer.dll
TRT_DLLS += $$TRT_PATH/nvonnxparser.dll
TRT_DLLS += $$TRT_PATH/nvinfer_plugin.dll

#cuDNN
CUDNN_PATH = "D:/code/cudnn-windows-x86_64-8.9.7.29_cuda12-archive"
CUDNN_INC = $$CUDNN_PATH/include
CUDNN_DLL_PATH = $$CUDNN_PATH/bin
CUDNN_LIB_PATH = $$CUDNN_PATH/lib/x64
CUDNN_DLLS += $$CUDNN_DLL_PATH/cudnn64_8.dll


INCLUDEPATH += $$CUDAINC
INCLUDEPATH += $$FASTVIDEO_INC
INCLUDEPATH += $$FASTVIDEOPATH/core_samples
INCLUDEPATH += $$FFMPEG_PATH/include
INCLUDEPATH += $$JPEGTURBO/include
INCLUDEPATH += $$ONNX_INC
INCLUDEPATH += $$ZJV_INC
#INCLUDEPATH += $$SNAP7_INC

#QMAKE_CXXFLAGS += "/WX" # Treats all compiler warnings as errors.
#QMAKE_LFLAGS_WINDOWS += "/NODEFAULTLIB:LIBCMT"

QT_DLLS += Qt"$$QT_MAJOR_VERSION"Core
QT_DLLS += Qt"$$QT_MAJOR_VERSION"Gui
QT_DLLS += Qt"$$QT_MAJOR_VERSION"Widgets
QT_DLLS += Qt"$$QT_MAJOR_VERSION"Svg
QT_DLLS += Qt"$$QT_MAJOR_VERSION"Network
QT_DLLS += Qt"$$QT_MAJOR_VERSION"Charts

#QT_DLLS += Qt"$$QT_MAJOR_VERSION"OpenGL
#QT_DLLS += Qt5Xml
#QT_DLLS += Qt5Multimedia


CONFIG(debug, debug|release){
    QMAKE_LFLAGS_WINDOWS += "/NODEFAULTLIB:LIBCMTD"
    QMAKE_LFLAGS_WINDOWS += "/NODEFAULTLIB:MSVCRT"
}else:CONFIG(release, debug|release){
    #For release build debug
    QMAKE_LFLAGS_WINDOWS += "/INCREMENTAL:NO /DEBUG /OPT:REF /OPT:ICF"
    QMAKE_CXXFLAGS += "/Zi /DEBUG"
}

#LIBS += $$FASTVIDEO_LIB
#LIBS += $$FASTVIDEO_EXTRA_LIBS
LIBS += -L$$FFMPEG_LIB  -lavcodec -lavformat -lavutil -lswresample
LIBS += -L$$CUDA_TOOLKIT_PATH/lib/$$PLATFORM -lcudart -lcuda -lcublas
#LIBS += -lglu32 -lopengl32 -lgdi32 -luser32 -lMscms -lShell32 -lOle32 -lWs2_32 -lstrmiids -lComdlg32
LIBS += -L$$JPEGTURBO/lib -ljpeg62_8bit
LIBS += -L$$ONNX_LIB  -lonnxruntime
LIBS += -L$$ZJV_LIB  -lzj_pipe
LIBS += -L$$TRT_DLL_PATH -lnvinfer -lnvonnxparser -lnvinfer_plugin
LIBS += -L$$CUDNN_LIB_PATH -lcudnn
LIBS += -L$$SNAP7_LIB  -lsnap7


contains(DEFINES, SUPPORT_XIMEA){
    XI_API_PATH = $$OTHER_LIB_PATH/XIMEA/API
    INCLUDEPATH += $$XI_API_PATH
    LIBS += -L$$XI_API_PATH/x64 -lxiapi64
    FASTVIDEO_EXTRA_DLLS += $$XI_API_PATH/x64/xiapi64.dll
}

contains(DEFINES, SUPPORT_FLIR){
    INCLUDEPATH += $$FLIR_PATH/include
    LIBS += -L$$FLIR_PATH/lib64/vs2015 -lSpinnaker_v140
    FASTVIDEO_EXTRA_DLLS += $$FLIR_PATH/bin64/vs2015/Spinnaker_v140.dll
}

contains(DEFINES, SUPPORT_IMPERX){
    INCLUDEPATH += $$IMPERX_PATH/inc
    LIBS += -L$$IMPERX_PATH/lib/win64_x64 -lIpxCameraApi
    FASTVIDEO_EXTRA_DLLS += $$IMPERX_PATH/bin/win64_x64/IpxCameraApi.dll
    FASTVIDEO_EXTRA_DLLS += $$IMPERX_PATH/bin/win64_x64/IpxCTI.cti
}

contains(DEFINES, SUPPORT_GENICAM){
    DEFINES -= UNICODE
    DEFINES += GENICAM_NO_AUTO_IMPLIB

    GENAPIPATH = $$OTHER_LIB_PATH/GenICam/library/CPP

    FASTVIDEO_EXTRA_DLLS += $${GENAPIPATH}/bin/Win64_x64/GCBase_MD_$${GENAPIVER}.dll
    FASTVIDEO_EXTRA_DLLS += $${GENAPIPATH}/bin/Win64_x64/GenApi_MD_$${GENAPIVER}.dll
    FASTVIDEO_EXTRA_DLLS += $${GENAPIPATH}/bin/Win64_x64/GenCP_MD_$${GENAPIVER}.dll
    FASTVIDEO_EXTRA_DLLS += $${GENAPIPATH}/bin/Win64_x64/log4cpp_MD_$${GENAPIVER}.dll
    FASTVIDEO_EXTRA_DLLS += $${GENAPIPATH}/bin/Win64_x64/Log_MD_$${GENAPIVER}.dll
    FASTVIDEO_EXTRA_DLLS += $${GENAPIPATH}/bin/Win64_x64/XmlParser_MD_$${GENAPIVER}.dll
    FASTVIDEO_EXTRA_DLLS += $${GENAPIPATH}/bin/Win64_x64/MathParser_MD_$${GENAPIVER}.dll
    FASTVIDEO_EXTRA_DLLS += $${GENAPIPATH}/bin/Win64_x64/NodeMapData_MD_$${GENAPIVER}.dll

    INCLUDEPATH += $$GENAPIPATH/include
    LIBS += -L$$GENAPIPATH/lib/Win64_x64 -lGCBase_MD_$${GENAPIVER} -lGenApi_MD_$${GENAPIVER} -lGenCP_MD_$${GENAPIVER}
    LIBS += -llog4cpp_MD_$${GENAPIVER} -lLog_MD_$${GENAPIVER} -lXmlParser_MD_$${GENAPIVER}
}
contains(DEFINES, SUPPORT_BASLER){
    BASLER_PATH = $$OTHER_LIB_PATH/Basler

    FASTVIDEO_EXTRA_DLLS += $${BASLER_PATH}/Runtime/$$PLATFORM/GCBase_MD_VC141_v3_1_Basler_pylon.dll
    FASTVIDEO_EXTRA_DLLS += $${BASLER_PATH}/Runtime/$$PLATFORM/PylonC_v7_1.dll

    INCLUDEPATH += $$BASLER_PATH/Development/include
    LIBS += -L$$BASLER_PATH/Development/lib/$$PLATFORM -lGCBase_MD_VC141_v3_1_Basler_pylon -lPylonC_v7_1
}
contains(DEFINES, SUPPORT_LUCID){

#    LUCID_ROOT = $$LUCID_DEV_ROOT
    LUCID_ROOT = "$${OTHER_LIB_PATH}/Arena SDK"
    LUCID_GENICAM = $${LUCID_ROOT}/GenICam

    INCLUDEPATH += $${LUCID_ROOT}/include/Arena
    INCLUDEPATH += $${LUCID_ROOT}/include/GenTL
    INCLUDEPATH += $${LUCID_GENICAM}/library/CPP/include

    LIBS += -L$${LUCID_ROOT}/lib64/Arena
    LIBS += -L$${LUCID_GENICAM}/library/CPP/lib/Win64_x64

CONFIG(debug, debug|release){
    SUF = "d"
    LUCID_DLL_PATH = $${LUCID_ROOT}/x64Debug
}else:CONFIG(release, debug|release){
    SUF = ""
    LUCID_DLL_PATH = $${LUCID_ROOT}/x64Release
}

    LIBS += -lArena$${SUF}_v140
    LIBS += -lGenTL_LUCID$${SUF}_v140
    LIBS += -lGCBase_MD$${SUF}_VC140_v3_3_LUCID
    LIBS += -lGenApi_MD$${SUF}_VC140_v3_3_LUCID
    LIBS += -lLog_MD$${SUF}_VC140_v3_3_LUCID
    LIBS += -lMathParser_MD$${SUF}_VC140_v3_3_LUCID
    LIBS += -lNodeMapData_MD$${SUF}_VC140_v3_3_LUCID
    LIBS += -lXmlParser_MD$${SUF}_VC140_v3_3_LUCID
    LIBS += -llucidlog$${SUF}_v140

    FASTVIDEO_EXTRA_DLLS += $${LUCID_DLL_PATH}\Arena$${SUF}_v140.dll
    FASTVIDEO_EXTRA_DLLS += $${LUCID_DLL_PATH}\GCBase_MD$${SUF}_VC140_v3_3_LUCID.dll
    FASTVIDEO_EXTRA_DLLS += $${LUCID_DLL_PATH}\GenApi_MD$${SUF}_VC140_v3_3_LUCID.dll
    FASTVIDEO_EXTRA_DLLS += $${LUCID_DLL_PATH}\GenTL_LUCID$${SUF}_v140.dll
    FASTVIDEO_EXTRA_DLLS += $${LUCID_DLL_PATH}\Log_MD$${SUF}_VC140_v3_3_LUCID.dll
    FASTVIDEO_EXTRA_DLLS += $${LUCID_DLL_PATH}\MathParser_MD$${SUF}_VC140_v3_3_LUCID.dll
    FASTVIDEO_EXTRA_DLLS += $${LUCID_DLL_PATH}\NodeMapData_MD$${SUF}_VC140_v3_3_LUCID.dll
    FASTVIDEO_EXTRA_DLLS += $${LUCID_DLL_PATH}\XmlParser_MD$${SUF}_VC140_v3_3_LUCID.dll
    FASTVIDEO_EXTRA_DLLS += $${LUCID_DLL_PATH}\lucidlog$${SUF}_v140.dll
    }

contains(DEFINES, USE_NV_API){
    NVAPI_PATH = "C:/Program Files (x86)/NVIDIA Corporation/Nsight Visual Studio Edition 5.6/Monitor/nvapi"
    INCLUDEPATH += $$NVAPI_PATH
    LIBS += -L$$NVAPI_PATH/amd64 -lnvapi64
}
