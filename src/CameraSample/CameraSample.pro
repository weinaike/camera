QT += core gui widgets network opengl charts

include(../common_defs.pri)
include(../common_funcs.pri)
win32: include(../common.pri)
unix:  include(../common_unix.pri)

TARGET = $$PROJECT_NAME
TEMPLATE = app

unix:  FASTVIDEO_EXTRA_DLLS += $$PWD/GPUCameraSample.sh

CONFIG += console
CONFIG += c++11

INCLUDEPATH += $$PWD
INCLUDEPATH += $$PWD/CUDASupport
INCLUDEPATH += $$PWD/Widgets
INCLUDEPATH += $$PWD/Camera
INCLUDEPATH += $$PWD/RtspServer
#INCLUDEPATH += "C:\Program Files\Lucid Vision Labs\Arena SDK\include\Arena"
#INCLUDEPATH += "C:\Program Files\Lucid Vision Labs\Arena SDK\GenICam\library\CPP\include"

SOURCES += main.cpp\
    Camera/BaslerCamera.cpp \
    MainWindow.cpp \
    Globals.cpp \
    AppSettings.cpp \
    Widgets/ImageResult.cpp \
    ppm.cpp \
    RawProcessor.cpp \
    AsyncFileWriter.cpp \
    MJPEGEncoder.cpp \
    avfilewriter/avfilewriter.cpp \
    Camera/GPUCameraBase.cpp \
    Camera/FrameBuffer.cpp \
    Camera/PGMCamera.cpp \
    CUDASupport/CUDAProcessorBase.cpp \
    CUDASupport/CUDAProcessorGray.cpp \
    Widgets/DenoiseController.cpp \
    Widgets/GtGWidget.cpp \
    Widgets/CameraSetupWidget.cpp \
    Widgets/camerastatistics.cpp \
    RtspServer/JpegEncoder.cpp \
    Widgets/customlabel.cpp \
    snap7.cpp


HEADERS  += MainWindow.h \
    Camera/BaslerCamera.h \
    Globals.h \
    AppSettings.h \
    Widgets/ImageResult.h \
    ppm.h \
    RawProcessor.h \
    AsyncFileWriter.h \
    AsyncQueue.h \
    MJPEGEncoder.h \
    avfilewriter/avfilewriter.h \
    Camera/GPUCameraBase.h \
    Camera/FrameBuffer.h \
    Camera/PGMCamera.h \
    CUDASupport/CUDAProcessorGray.h \
    CUDASupport/CUDAProcessorBase.h \
    CUDASupport/CUDAProcessorOptions.h \
    CUDASupport/CudaAllocator.h \
    CUDASupport/GPUImage.h \
    Widgets/DenoiseController.h \
    Widgets/camerastatistics.h \
    Widgets/GtGWidget.h \
    Widgets/CameraSetupWidget.h \
    RtspServer/common_utils.h \
    RtspServer/JpegEncoder.h \
    version.h \
    Widgets/customlabel.h \
    snap7.h

win32{
    SOURCES +=  $$PWD/../../OtherLibs/FastvideoSDK/common/BaseAllocator.cpp \
                $$PWD/../../OtherLibs/FastvideoSDK/common/SurfaceTraits.cpp \
                $$PWD/../../OtherLibs/FastvideoSDK/common/alignment.cpp \
                $$PWD/../../OtherLibs/FastvideoSDK/core_samples/SurfaceTraitsInternal.cpp
#               $$FASTVIDEO_SDK/common/FastAllocator.cpp

    HEADERS +=  $$PWD/../../OtherLibs/FastvideoSDK/common/BaseAllocator.h \
                $$PWD/../../OtherLibs/FastvideoSDK/common/SurfaceTraits.hpp \
                $$PWD/../../OtherLibs/FastvideoSDK/common/alignment.hpp \
                $$PWD/../../OtherLibs/FastvideoSDK/core_samples/SurfaceTraitsInternal.hpp
#               $$FASTVIDEO_SDK/common/FastAllocator.h

}else{
    SOURCES += $$PWD/../../OtherLibsLinux/FastvideoSDK/common/BaseAllocator.cpp \
               $$PWD/../../OtherLibsLinux/FastvideoSDK/common/FastAllocator.cpp

    HEADERS += $$PWD/../../OtherLibsLinux/FastvideoSDK/common/BaseAllocator.h \
               $$PWD/../../OtherLibsLinux/FastvideoSDK/common/FastAllocator.h
}




contains(DEFINES, ENABLE_GL){
   SOURCES += Widgets/GLImageViewer.cpp
   HEADERS += Widgets/GLImageViewer.h
}


contains(DEFINES, SUPPORT_XIMEA ){
   SOURCES += Camera/XimeaCamera.cpp
   HEADERS += Camera/XimeaCamera.h
}

contains(DEFINES, SUPPORT_FLIR ){
   SOURCES += Camera/FLIRCamera.cpp
   HEADERS += Camera/FLIRCamera.h
}

contains(DEFINES, SUPPORT_IMPERX ){
   SOURCES += Camera/ImperxCamera.cpp
   HEADERS += Camera/ImperxCamera.h
}

contains(DEFINES, SUPPORT_GENICAM ){
    SOURCES += rc_genicam_api/buffer.cc \
    rc_genicam_api/config.cc \
    rc_genicam_api/cport.cc \
    rc_genicam_api/device.cc \
    rc_genicam_api/exception.cc \
    rc_genicam_api/image.cc \
    rc_genicam_api/imagelist.cc \
    rc_genicam_api/interface.cc \
    rc_genicam_api/pointcloud.cc \
    rc_genicam_api/stream.cc \
    rc_genicam_api/system.cc \
    Camera/GeniCamCamera.cpp

    unix:  SOURCES += rc_genicam_api/gentl_wrapper_linux.cc
    win32: SOURCES += rc_genicam_api/gentl_wrapper_win32.cc

    HEADERS  +=  rc_genicam_api/buffer.h \
    rc_genicam_api/config.h \
    rc_genicam_api/cport.h \
    rc_genicam_api/device.h \
    rc_genicam_api/exception.h \
    rc_genicam_api/gentl_wrapper.h \
    rc_genicam_api/image.h \
    rc_genicam_api/imagelist.h \
    rc_genicam_api/interface.h \
    rc_genicam_api/pixel_formats.h \
    rc_genicam_api/pointcloud.h \
    rc_genicam_api/stream.h \
    rc_genicam_api/system.h \
    Camera/GeniCamCamera.h
}

contains(DEFINES, SUPPORT_LUCID ){
   SOURCES += Camera/LucidCamera.cpp
   HEADERS += Camera/LucidCamera.h
}

contains(TARGET_ARCH, arm64 ) {
    contains(DEFINES, SUPPORT_MIPI){
        HEADERS += Camera/MIPICamera.h
        SOURCES += Camera/MIPICamera.cpp
    }
}else{
    DEFINES -= SUPPORT_MIPI
}

FORMS    += MainWindow.ui \
    Widgets/DenoiseController.ui \
    Widgets/CameraSetupWidget.ui \
    Widgets/ImageResult.ui \
    Widgets/camerastatistics.ui


RC_FILE = gpu-camera-sample.rc
#resource.rc

unix:copySelectedPluginsToDestdir($$QT_SELECTED_PLUGIN)

#copyPluginsToDestdir(audio)
#win32:copyPluginsToDestdir(sqldrivers)
#copyPluginsToDestdir(printsupport)
win32:copyPluginsToDestdir(platforms)
win32:copyPluginsToDestdir(imageformats)

copyQtDllsToDestdir($$QT_DLLS)
copyToDestdir($$FASTVIDEO_EXTRA_DLLS)
copyToDestdir($$TRT_DLL)


unix {

    contains(TARGET_ARCH, arm64){
        include($$PWD/jetson_api/jetson_api.pri)
    }

    copyPluginsToDestdir(xcbglintegrations)
    copyQtIcuDllsToDestdir($$QT_ICU_DLLS)
    makeLinks()
}

win32 {
    copyToDestdir($$VC_REDIST_DLLS)
    for(ifile, QT_EXECS) {
        copyToDestdir($$[QT_INSTALL_BINS]/$$ifile)
    }

    copyToDestdir($$FASTVIDEO_DLL)
    copyToDestdir($$CUDA_DLL)
}


RESOURCES += \
    Resorces.qrc

DISTFILES +=

