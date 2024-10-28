
#ifndef ASYNCCONTROL_H
#define ASYNCCONTROL_H

#include <QObject>
#include <QWaitCondition>
#include <QMutex>
#include <QThread>
#include "AsyncQueue.h"
#include "snap7.h"
#include "WeldData.h"

struct LaserControlData
{
    int     command_id;     // command id    
    float   power;          // Laser power
    float   de_focus;       // De-focus
    float   speed;          // Welding speed
};


class AsyncControl : public QObject
{
    Q_OBJECT
public:
    explicit AsyncControl(int maxSize, QObject *parent = nullptr);
    ~AsyncControl();

    int connect(const char * ip, int rock, int slot);
    int start();
    int disconnect();
    bool get_connect_status(){return plcClient->Connected();}
    int  put(WeldResult task, int db = 0);
    void wake(){mStart.wakeAll();}
    int  queueSize(){return mQueue.count();}
    int  stop();
    bool  get_status(){return mControlling;}


protected:
    void startControl();
    // 返回是否发送调整信号， 
    bool judagePower(std::vector<WeldResult> &result, float &power_ratio);

    bool mCancel  = false ;
    bool mControlling = false;
    int  mDBID = 0;
    QMutex mLock;
    QWaitCondition mStart;
    QThread mWorkThread;
    AsyncQueue<WeldResult> mQueue;
    int maxQueuSize = 32;
    int mDropped = 0;
    TS7Client * plcClient = nullptr;
};


#endif // ASYNCCONTROL_H
