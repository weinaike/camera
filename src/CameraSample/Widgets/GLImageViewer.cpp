/*
 Copyright 2011-2019 Fastvideo, LLC.
 All rights reserved.

 This file is a part of the GPUCameraSample project
 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 3. Any third-party SDKs from that project (XIMEA SDK, Fastvideo SDK, etc.) are licensed on different terms. Please see their corresponding license terms.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 The views and conclusions contained in the software and documentation are those
 of the authors and should not be interpreted as representing official policies,
 either expressed or implied, of the FreeBSD Project.
*/

#include "GLImageViewer.h"
#include <QApplication>
#include <QMouseEvent>
#include <QPoint>
#include <QScreen>
#include <QTimer>

#include <cmath>
#include <GL/gl.h>


namespace
{
    const qreal zoomStep = 0.1;
    const qreal zoomMin = 0.1;
    const qreal zoomMax = 8.0;
}

GLImageViewer::GLImageViewer(GLRenderer *renderer) :
    QOpenGLWindow(),
    mRenderer(renderer)
{
    setSurfaceType(QWindow::OpenGLSurface);
    setFormat(renderer->format());

    mZoom = 1.;
    mPtDown = QPoint(-1, -1);
    mTexTopLeft = QPoint(0, 0);
    mViewMode = GLImageViewer::vmZoomFit;
    mShowImage = false;
}

GLImageViewer::~GLImageViewer(){}

void GLImageViewer::clear()
{
    if(mRenderer != nullptr)
        mRenderer->showImage(false);
    update();
}

void GLImageViewer::load(void* img, int width, int height)
{

    if(mRenderer == nullptr)
        return;
    //mImageSize = QSize(width, height);
    //mShowImage = true;


    mRenderer->loadImage(img, width, height);
    setViewMode(mViewMode);
    update();
}

void GLImageViewer::setViewMode(GLImageViewer::ViewMode mode)
{
    mViewMode = mode;
    if(mViewMode == vmZoomFit)
    {
        setFitZoom(size());
    }
    update();
}


GLImageViewer::ViewMode GLImageViewer::getViewMode() const
{
    return mViewMode;
}


void GLImageViewer::setZoom(qreal scale)
{
    setZoomInternal(scale);
}

void GLImageViewer::setZoomInternal(qreal newZoom, QPoint fixPoint)
{
    if(newZoom < ::zoomMin || newZoom > ::zoomMax || newZoom == getZoom())
        return;

    qreal zoom = getZoom();

    if(fixPoint.isNull())
        fixPoint = geometry().center();

    float x = fixPoint.x();
    float y = fixPoint.y();

    mTexTopLeft += QPointF(x * (1. / zoom - 1. / newZoom), -(height() - y) * (1. / zoom - 1. / newZoom));

    mZoom = newZoom;
    adjustTexTopLeft();
    update();
    emit zoomChanged(newZoom);
}

void GLImageViewer::adjustTexTopLeft()
{
    float w = width();
    float h = height();

    QSize imageSize(mRenderer->imageSize());
    float iw = imageSize.width();
    float ih = imageSize.height();

    if(mTexTopLeft.x() < 0)
        mTexTopLeft.setX(0);

    if(iw - w / mZoom > 0)
    {
        if(mTexTopLeft.x() > iw - w / mZoom)
            mTexTopLeft.setX(iw - w / mZoom);
    }

    if(mTexTopLeft.y() < h / mZoom)
        mTexTopLeft.setY(h / mZoom);

    if(mTexTopLeft.y() > ih)
        mTexTopLeft.setY(ih);

}

qreal GLImageViewer::getZoom() const
{
    return mZoom;
}

void GLImageViewer::resizeEvent(QResizeEvent * event)
{
    if(mViewMode == vmZoomFit)
    {
        QSize sz = event->size();

        setFitZoom(sz);
        emit sizeChanged(sz);
    }
    QOpenGLWindow::resizeEvent(event);
}

void GLImageViewer::mouseMoveEvent(QMouseEvent* event)
{
    if(event->buttons() != Qt::LeftButton)
        return;
    if(mPtDown.isNull())
        return;

    float dx = mPtDown.x() - event->pos().x();
    float dy = mPtDown.y() - event->pos().y();

    mTexTopLeft.rx() = mTexTopLeft.x() + dx / mZoom;
    mTexTopLeft.ry() = mTexTopLeft.y() + dy / mZoom;

    adjustTexTopLeft();
    update();

    mPtDown = event->pos();
}

void GLImageViewer::mousePressEvent(QMouseEvent* event)
{
    if(event->buttons() == Qt::LeftButton)
        mPtDown = event->pos();
    else
        emit contextMenu(event->globalPos());
}

void GLImageViewer::mouseReleaseEvent(QMouseEvent* event)
{
    Q_UNUSED(event)
    mPtDown = QPoint();

    if(currentTool == tlWBPicker)
    {
        QPoint pt(screenToBitmap(event->pos() * screen()->devicePixelRatio()));
            emit newWBFromPoint(pt);
    }

}

void GLImageViewer::setFitZoom(QSize szClient)
{
    if(!mRenderer)
        return;

    szClient -=(QSize(6,6));
    QSize imageSize(mRenderer->imageSize());

    if(imageSize.isEmpty())
        return;

    qreal zoom = qMin((qreal)(szClient.height()) / (qreal)(imageSize.height()), (qreal)(szClient.width()) / (qreal)(imageSize.width()));
    setZoom(zoom);
}

void GLImageViewer::setCurrentTool(const Tool &tool)
{
    currentTool = tool;
    if(mRenderer)
        mRenderer->update();
}

void GLImageViewer::wheelEvent(QWheelEvent * event)
{
    if(mViewMode == vmZoomFit)
        return;

    float numDegrees = event->angleDelta().x() / 8.;
    float numSteps = numDegrees / 15.;

    Qt::KeyboardModifiers keyState = QApplication::queryKeyboardModifiers();
    if(keyState.testFlag(Qt::ControlModifier))
    {
        qreal newZoom = getZoom() * std::pow(1.125, numSteps);
#if QT_VERSION_MAJOR >= 6
        QPointF p(event->position());
        setZoomInternal(newZoom,  QPoint(p.x(), p.y()));
#else
        setZoomInternal(newZoom, event->pos());
#endif
        update();
    }
}

void GLImageViewer::exposeEvent(QExposeEvent *event)
{
    Q_UNUSED(event);
    update();
}

QPoint GLImageViewer::screenToBitmap(const QPoint& pt)
{
    if(!mRenderer)
        return QPoint();

    QSize imageSize(mRenderer->imageSize());

    qreal w = width();
    qreal h = height();

    qreal iw = imageSize.width();
    qreal ih = imageSize.height();

    qreal dx = 0.;
    qreal dy = 0.;
    if(iw < w / mZoom)
        dx = (w / mZoom - iw) / 2;

    if(ih < h / mZoom)
        dy = (h / mZoom - ih) / 2;

    QPoint ret = QPoint(int(mTexTopLeft.x() + pt.x() / mZoom - dx),
                        int((mTexTopLeft.y() - (h - pt.y()) / mZoom) + dy));

    return ret;
}

QPoint GLImageViewer::bitmapToScreen(const QPoint& pt)
{
    if(!mRenderer)
        return QPoint();

    QSize imageSize(mRenderer->imageSize());

    qreal w = width();
    qreal h = height();

    qreal iw = imageSize.width();
    qreal ih = imageSize.height();

    qreal dx = 0.;
    qreal dy = 0.;
    if(iw - w / mZoom < 0)
        dx = (w - iw * mZoom) / 2;

    if(ih - h / mZoom < 0)
        dy = (h - ih * mZoom) / 2;

    return QPoint(int((pt.x() - mTexTopLeft.x()) * mZoom + dx),
                  int((pt.y() - mTexTopLeft.y()) * mZoom + dy));
}


GLRenderer::GLRenderer(QObject *parent):
    QObject(parent)

{
    m_format.setDepthBufferSize(16);
    m_format.setSwapInterval(1);
    m_format.setSwapBehavior(QSurfaceFormat::DoubleBuffer);
    m_format.setRenderableType(QSurfaceFormat::OpenGL);
    m_format.setProfile(QSurfaceFormat::CoreProfile);

    m_context = new QOpenGLContext(this);
    m_context->setFormat(m_format);
    m_context->create();

#ifndef __aarch64__
    mRenderThread.setObjectName(QStringLiteral("RenderThread"));
    moveToThread(&mRenderThread);
    mRenderThread.start();
#endif
}

GLRenderer::~GLRenderer()
{
#ifndef __aarch64__
    mRenderThread.quit();
    mRenderThread.wait(3000);
#endif
}

void GLRenderer::initialize()
{
    QSize sz = QGuiApplication::primaryScreen()->size() * 2;
    initializeOpenGLFunctions();

    GLint bsize;
    glGenTextures(1, &texture);
    glGenBuffers(1, &pbo_buffer);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, GLsizeiptr(3 * sizeof(unsigned char)) * sz.width() * sz.height(), nullptr, GL_STREAM_COPY);
    glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    m_program = new QOpenGLShaderProgram(this);
    m_program->addShaderFromSourceFile(QOpenGLShader::Vertex, ":/quadVertex.vert");
    m_program->addShaderFromSourceFile(QOpenGLShader::Fragment, ":/quadFragment.frag");
    m_program->link();

    m_texUniform = m_program->uniformLocation("tex");
    m_vertPosAttr = m_program->attributeLocation("vertexPosAttr");
    m_texPosAttr = m_program->attributeLocation("texPosAttr");

    glDisable(GL_TEXTURE_2D);
}

void GLRenderer::showImage(bool show)
{
    mShowImage = show;
}

void GLRenderer::update()
{
    QTimer::singleShot(0, this, [this](){render();});
}

void GLRenderer::render()
{
    if(mRenderWnd == nullptr)
        return;

    if(!mRenderWnd->isExposed())
        return;

    if(!m_context->makeCurrent(mRenderWnd))
        return;

    if(!m_initialized)
    {
        initialize();
        m_initialized = true;
    }

    if(mImageSize.isEmpty() || !mShowImage)
    {
        //Render empty background
        glViewport(0, 0, 1, 1);
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(0, 1, 0, 1, 0, 1);
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glClearColor(0.25, 0.25, 0.25, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        m_context->swapBuffers(mRenderWnd);
        m_context->doneCurrent();

        return;
    }

    //qDebug("mImageSize.isEmpty() = %d, mShowImage = %d", mImageSize.isEmpty(), mShowImage);

    int w = mRenderWnd->width();
    int h = mRenderWnd->height();

    GLfloat iw = mImageSize.width();
    GLfloat ih = mImageSize.height();
    // 设置视口和投影矩阵来渲染图像
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 1, 0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    // 设置清除颜色并清除颜色缓冲区
    glClearColor(0.25, 0.25, 0.25, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    // 禁用深度测试
    glDisable(GL_DEPTH_TEST);
    // 激活和绑定纹理
    glActiveTexture(GL_TEXTURE1);
    glEnable(GL_TEXTURE_2D);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
    glBindTexture(GL_TEXTURE_2D, texture);
    // 将纹理图像加载到 GPU
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, iw, ih, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

    // 计算纹理坐标和矩形顶点
    float rectLeft, rectTop, rectRight, rectBottom;
    float texLeft, texTop, texRight, texBottom;

    //Calc left and right bounds
    auto zoom = GLfloat(mRenderWnd->getZoom());

    QPointF texTopLeft(mRenderWnd->texTopLeft());
    if(w / zoom <= iw)
    {
        rectLeft = 0;
        rectRight = 1;
        texLeft = texTopLeft.x();

        texRight = texLeft + w / zoom;
        if(texRight > iw)
            texRight = iw;
    }
    else
    {
        rectLeft = (w - iw * zoom) / (2 * w);
        rectRight = rectLeft + (iw * zoom) / w;
        texLeft = 0;
        texRight = iw;
    }

    //Calc top and bottom bounds
    if(h / zoom <= ih)
    {
        rectTop = 1;
        rectBottom = 0;
        texBottom =  texTopLeft.y() - h / zoom;
        if(texBottom < 0)
            texBottom = 0;
        texTop = texBottom + h / zoom;
        if(texTop > ih)
            texTop = ih;
    }
    else
    {
        rectBottom = (h - ih * zoom) / (2 * h);
        rectTop = rectBottom + (ih * zoom) / h;

        texBottom = 0;
        texTop = ih;
    }
    texLeft /= iw;
    texRight /= iw;

    texTop /= ih;
    texBottom /= ih;
    // 绑定着色器程序并设置纹理
    m_program->bind();
    m_program->setUniformValue(m_texUniform, texture);
    glDisable(GL_TEXTURE_2D);
    // 调整矩形坐标到 -1...1 坐标系统
    //Adjust calculated rectangle to -1...1 coordinate system
    rectRight = 2 * rectRight - 1;
    rectLeft = 2 * rectLeft - 1;
    rectTop = 2 * rectTop - 1;
    rectBottom = 2 * rectBottom - 1;
    // 定义顶点和纹理坐标
    GLfloat vertices[] = {
        rectRight, rectTop,
        rectRight, rectBottom,
        rectLeft, rectBottom,
        rectLeft, rectBottom,
        rectLeft, rectTop,
        rectRight, rectTop
    };

    GLfloat texCoords[] = {
        texRight, texBottom,
        texRight, texTop,
        texLeft, texTop,
        texLeft, texTop,
        texLeft, texBottom,
        texRight, texBottom
    };
    // 设置顶点属性指针并启用顶点属性数组
    glVertexAttribPointer(m_vertPosAttr, 2, GL_FLOAT, GL_FALSE, 0, vertices);
    glVertexAttribPointer(m_texPosAttr, 2, GL_FLOAT, GL_FALSE, 0, texCoords);

    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    // 绘制矩形（两个三角形组成）
    glDrawArrays(GL_TRIANGLES, 0, 6);
    // 禁用顶点属性数组
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(0);
    // 释放着色器程序
    m_program->release();
    // 解绑纹理和 PBO
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    // 完成渲染并交换缓冲区
    glFinish();
    m_context->swapBuffers(mRenderWnd);
    glFinish();
}

void GLRenderer::loadImage(void* img, int width, int height)
{
    QTimer::singleShot(0, this, [this, img,width,height](){loadImageInternal(img, width, height);});
}

void GLRenderer::loadImageInternal(void* img, int width, int height)
{
    unsigned char *data = NULL;
    size_t pboBufferSize = 0;

    cudaError_t error = cudaSuccess;
    // 设置内部图像大小变量
    mImageSize = QSize(width, height);
    // 确保 OpenGL 上下文是活动的
    if(!m_context->makeCurrent(mRenderWnd))
        return;
    // 如果未初始化 PBO，则进行初始化
    if(!pbo_buffer)
    {
        initialize();
        m_initialized = true;
    }
    // 绑定 PBO 并设置其参数
    GLint bsize;
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    // 分配 PBO 内存
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 3 * sizeof(unsigned char) * width * height, NULL, GL_STREAM_COPY);
    glGetBufferParameteriv(GL_PIXEL_UNPACK_BUFFER, GL_BUFFER_SIZE, &bsize);
    struct cudaGraphicsResource* cuda_pbo_resource = 0;
    // 如果 img 为空，则返回
    if(img == nullptr)
        return;
    // 将 OpenGL PBO 注册为 CUDA 图形资源
    error = cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo_buffer, cudaGraphicsMapFlagsWriteDiscard);
    if(error != cudaSuccess)
    {
        qDebug("Cannot register CUDA Graphic Resource: %s\n", cudaGetErrorString(error));
        return;
    }
    // 映射 CUDA 资源
    if((error = cudaGraphicsMapResources( 1, &cuda_pbo_resource, 0 ) ) != cudaSuccess)
    {
        qDebug("cudaGraphicsMapResources failed: %s\n", cudaGetErrorString(error) );
        return;
    }
    // 获取映射的 CUDA 资源指针
    if((error = cudaGraphicsResourceGetMappedPointer( (void **)&data, &pboBufferSize, cuda_pbo_resource ) ) != cudaSuccess )
    {
        qDebug("cudaGraphicsResourceGetMappedPointer failed: %s\n", cudaGetErrorString(error) );
        return;
    }
    // 检查 PBO 缓冲区大小
    if(pboBufferSize < ( width * height * 3 * sizeof(unsigned char) ))
    {
        qDebug("cudaGraphicsResourceGetMappedPointer failed: %s\n", cudaGetErrorString(error) );
        return;
    }
    // 将图像数据复制到 PBO
    if((error = cudaMemcpy( data, img, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToDevice ) ) != cudaSuccess)
    {
        qDebug("cudaMemcpy failed: %s\n", cudaGetErrorString(error) );
        return;
    }
    // 取消映射 CUDA 资源
    if((error = cudaGraphicsUnmapResources( 1, &cuda_pbo_resource, 0 ) ) != cudaSuccess )
    {
         qDebug("cudaGraphicsUnmapResources failed: %s\n", cudaGetErrorString(error) );
         return;
    }
     // 注销 CUDA 图形资源
    if(cuda_pbo_resource)
    {
        if((error  = cudaGraphicsUnregisterResource(cuda_pbo_resource))!= cudaSuccess)
        {
            qDebug("Cannot unregister CUDA Graphic Resource: %s\n", cudaGetErrorString(error));
            return;
        }
        cuda_pbo_resource = 0;
    }
    // 将 PBO 数据加载到 OpenGL 纹理
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo_buffer);
    glBindTexture(GL_TEXTURE_2D, texture);
    // 颜色格式：代码中假设图像是以 RGB 格式存储的，这可以从 glTexImage2D 函数调用中的参数 GL_RGB 看出。在 RGB 格式中，每个像素由三个颜色分量组成：红色、绿色和蓝色。
    // 像素大小：由于每个像素有三个颜色分量，且每个分量由 unsigned char 表示，因此每个像素占用 3 字节（3 * sizeof(unsigned char)）。这意味着 img 指针指向的内存区域的大小应该是 width * height * 3 字节。
    // 数据布局：img 中的像素数据应该是连续存储的，没有额外的填充或间隔。这是因为代码中使用 cudaMemcpy 直接从 img 指针指向的内存位置复制数据，且假设数据是紧密排列的。
    // 以上要求， img 的大小必须是 width * height * 3 字节，且数据应该按照标准的 24 位 RGB 格式排列（即每个像素由三个连续的 unsigned char 值组成
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glBindTexture(GL_TEXTURE_2D, 0);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
     // 释放 OpenGL 上下文
    m_context->doneCurrent();
}
