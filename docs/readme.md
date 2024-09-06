


## 代码目录结构

src
├── AppSettings.cpp
├── AppSettings.h
├── AsyncFileWriter.cpp
├── AsyncFileWriter.h
├── AsyncQueue.h
├── avfilewriter			// 视频文件写入类
│   ├── avfilewriter.cpp
│   └── avfilewriter.h
├── Camera				// 视频解析与相机驱动
│   ├── FrameBuffer.cpp
│   ├── FrameBuffer.h		// 数据缓存队列
│   ├── GeniCamCamera.cpp
│   ├── GeniCamCamera.h
│   ├── GPUCameraBase.cpp		// 基类
│   ├── GPUCameraBase.h
│   ├── LucidCamera.cpp		// 项目使用的相机驱动
│   ├── LucidCamera.h
│   ├── PGMCamera.cpp		// 处理视频文件
│   ├── PGMCamera.h
├── CameraSample.pro			// qt项目配置文件
├── CMakeLists.txt				// cmake项目配置文件
├── CUDASupport				// 图像处理类，主要处理图像变换，主要做图像预处理
│   ├── CudaAllocator.h
│   ├── CUDAProcessorBase.cpp
│   ├── CUDAProcessorBase.h
│   ├── CUDAProcessorGray.cpp
│   ├── CUDAProcessorGray.h
│   ├── CUDAProcessorOptions.h
│   ├── GPUImage.h
│   ├── gray12torgb8.cu
│   └── gray12torgb8.h
├── main.cpp
├── MainWindow.cpp			// 主窗口
├── MainWindow.h
├── MainWindow.ui
├── MJPEGEncoder.cpp			// 视频编码
├── MJPEGEncoder.h
├── ppm.cpp					// ppm格式输出
├── ppm.h
├── RawProcessor.cpp			// raw数据处理流程，其中会按顺序将数据发送给数据处理，数据显示， 模型推理， 结果显示模块。
├── RawProcessor.h
├── rc_genicam_api/ 		//通用相机通信协议接口
├── res/					//界面显示资源
├── RtspServer/			//图片编码
│   ├── common_utils.h
│   ├── JpegEncoder.cpp
│   └── JpegEncoder.h
└── Widgets/				//窗口类
    ├── GLImageViewer.cpp	//GL显示类
    ├── GLImageViewer.h
    ├── ImageResult.cpp		// 显示结果
    ├── ImageResult.h
    └── ImageResult.ui
