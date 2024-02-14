cmake_minimum_required(VERSION 3.18)

if(USE_CUDA)
    project(CameraSample VERSION 1.0 LANGUAGES CUDA)
else()
    project(CameraSample VERSION 1.0 LANGUAGES)
endif()

include_directories(${ADDITIONAL_INCDIR}
                    avfilewriter
                    Camera
                    CUDASupport
                    RtspServer
                    Widgets
                    .
                )


set(SRC
    AppSettings.cpp
    AsyncFileWriter.cpp
    Globals.cpp
    main.cpp
    MainWindow.cpp
    MJPEGEncoder.cpp
    ppm.cpp
    RawProcessor.cpp
    quadFragment.frag
    AppSettings.h
    AsyncFileWriter.h
    AsyncQueue.h
    Globals.h
    MainWindow.h
    MJPEGEncoder.h
    ppm.h
    RawProcessor.h
    resource.h
    version.h
    Resorces.qrc
    #gpu-camera-sample.rc
    resource.rc
    MainWindow.ui
    quadVertex.vert
    avfilewriter/avfilewriter.cpp
    avfilewriter/avfilewriter.h
    Camera/FrameBuffer.cpp
    Camera/FrameBuffer.h
    Camera/GPUCameraBase.cpp
    Camera/GPUCameraBase.h
    Camera/PGMCamera.cpp
    Camera/PGMCamera.h
    Camera/XimeaCamera.cpp
    Camera/XimeaCamera.h
    CUDASupport/CudaAllocator.h
    CUDASupport/CUDAProcessorBase.cpp
    CUDASupport/CUDAProcessorBase.h
    CUDASupport/CUDAProcessorGray.cpp
    CUDASupport/CUDAProcessorGray.h
    CUDASupport/CUDAProcessorOptions.h
    CUDASupport/GPUImage.h
    RtspServer/common_utils.h
    RtspServer/JpegEncoder.cpp
    RtspServer/JpegEncoder.h
    Widgets/CameraSetupWidget.cpp
    Widgets/CameraSetupWidget.h
    Widgets/CameraSetupWidget.ui
    Widgets/camerastatistics.cpp
    Widgets/camerastatistics.h
    Widgets/camerastatistics.ui
    Widgets/DenoiseController.cpp
    Widgets/DenoiseController.h
    Widgets/DenoiseController.ui
    Widgets/GtGWidget.cpp
    Widgets/GtGWidget.h
)
if(USE_CUDA)
    set(SRC ${SRC}
        Widgets/GLImageViewer.cpp
        Widgets/GLImageViewer.h
    )
endif()


set(SRC ${SRC}
    ../../${FASTLIB_DIR}/common/alignment.cpp
    ../../${FASTLIB_DIR}/common/alignment.hpp
    ../../${FASTLIB_DIR}/common/BaseAllocator.cpp
    ../../${FASTLIB_DIR}/common/BaseAllocator.h
#    ../../${FASTLIB_DIR}/common/FastAllocator.cpp
#    ../../${FASTLIB_DIR}/common/FastAllocator.h
    ../../${FASTLIB_DIR}/common/SurfaceTraits.cpp
    ../../${FASTLIB_DIR}/common/SurfaceTraits.hpp
    ../../${FASTLIB_DIR}/core_samples/SurfaceTraitsInternal.cpp
    ../../${FASTLIB_DIR}/core_samples/SurfaceTraitsInternal.hpp
)

if(${ARCHITECTURE} STREQUAL "aarch64")
    set(TEGRA_ARMABI aarch64-linux-gnu)

    find_library(NVBUF_LIB
        NAMES nvbuf_utils
        PATHS /usr/lib/$${TEGRA_ARMABI}/ /usr/lib/${TEGRA_ARMABI}/tegra/
        REQUIRED)
    find_library(V4L2_LIB
        NAMES v4l2 REQUIRED)
    list(APPEND ADDITIONAL_LIBS
        ${NVBUF_LIB}
        ${V4L2_LIB})

    include_directories(jetson_api)
    SET(SRC ${SRC}
        jetson_api/common_types.h
        jetson_api/jetson_api.pri
        jetson_api/nvbuf_utils.h
        jetson_api/nvvideoencoder.cpp
        jetson_api/nvvideoencoder.h
        jetson_api/v4l2encoder.cpp
        jetson_api/v4l2encoder.h
        jetson_api/v4l2_nv_extensions.h
        )
endif()

if(WIN32)
    add_executable(CameraSample WIN32 ${SRC})
else()
    if(USE_CUDA)
        add_executable(CameraSample ${SRC} CUDASupport/gray12torgb8.cu)
        set_target_properties(CameraSample PROPERTIES
        CXX_STANDARD 11
        CUDA_STANDARD 11
        )
    else()
        add_executable(CameraSample ${SRC})
        set_target_properties(CameraSample PROPERTIES
        CXX_STANDARD 11
        )
    endif()

endif()

if(USE_CUDA)   
    target_link_libraries(CameraSample PRIVATE
        Qt5::Core Qt5::Widgets Qt5::Gui Qt5::OpenGL Qt5::Network
        ${FFMPEG_LIB}
        ${JPEG_LIB}
        CUDA::cudart
        ${ADDITIONAL_LIBS}
    )
else()
#    target_link_libraries(CameraSample PRIVATE
#        Qt5::Core Qt5::Widgets Qt5::Gui Qt5::Network
#        ${FFMPEG_LIB}
#        ${JPEG_LIB}
#        ${ADDITIONAL_LIBS}
#    )

    target_link_libraries(CameraSample PRIVATE
        Qt5::Core Qt5::Widgets Qt5::Gui Qt5::Network
        ${FastVideo_LIB}
        ${FFMPEG_LIB}
        ${JPEG_LIB}
        ${ADDITIONAL_LIBS}
    )

endif()

if(WIN32)
    set(QT_QMAKE_EXECUTABLE "C:/Qt/Qt5.8.0/5.8/msvc2015_64/bin/qmake.exe" CACHE FILEPATH "Qt qmake executable" FORCE)
    get_filename_component(QTBINPATH ${QT_QMAKE_EXECUTABLE} DIRECTORY)

    add_custom_command(
        TARGET CameraSample POST_BUILD
        COMMAND ${QTBINPATH}/windeployqt ${CMAKE_BINARY_DIR}/bin/CameraSample.exe
    )
endif()

#add_custom_command(
#    TARGET CameraSample POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:Qt5::Widgets> ${CMAKE_BINARY_DIR}/bin/
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:Qt5::Core> ${CMAKE_BINARY_DIR}/bin/
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:Qt5::Gui> ${CMAKE_BINARY_DIR}/bin/
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:Qt5::Network> ${CMAKE_BINARY_DIR}/bin/
#    COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        $<TARGET_FILE:Qt5::OpenGL> ${CMAKE_BINARY_DIR}/bin/
#)
