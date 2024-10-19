#ifndef LUCIDCAMERA_H
#define LUCIDCAMERA_H

#ifdef SUPPORT_LUCID

#include "GPUCameraBase.h"
#include "ArenaApi.h"
#include <QObject>

class LucidCamera : public GPUCameraBase
{
public:
    LucidCamera();
    ~LucidCamera();

    virtual bool open(int devID);
    virtual bool start();
    virtual bool stop();
    virtual void close();

    virtual bool getParameter(cmrCameraParameter param, float& val);
    virtual bool setParameter(cmrCameraParameter param, float val);
    virtual bool getParameterInfo(cmrParameterInfo& info);


    int WriteStreamables(std::string);
    int ReadStreamables(std::string);
protected:

private:
    bool mStreaming = false;
    void startStreaming();
    Arena::ISystem* mSystem = nullptr;
    Arena::IDevice* mDevice = nullptr;

    void UpdateStatistics(Arena::IImage *pImage);


};

#endif // SUPPORT_LUCID

#endif // LUCIDCAMERA_H
