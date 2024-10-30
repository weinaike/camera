


#include <QTimer>
#include <QTime>
#include "AsyncControl.h"
#include <QMessageBox>
AsyncControl::AsyncControl(int maxSize, QObject *parent):
    QObject(parent), mQueue(maxSize)
{
    mWorkThread.setObjectName(QStringLiteral("laser Control Thread"));
    moveToThread(&mWorkThread);

    mWorkThread.start();
    maxQueuSize = maxSize;
    plcClient = new TS7Client();
    // 追加
    mLog.open("laser_control.log", std::ios::out | std::ios::app);
}

AsyncControl::~AsyncControl()
{
    disconnect();
    if(plcClient)
    {        
        delete plcClient;
        plcClient = nullptr;
    }   
    mLog.close();
    mWorkThread.quit();
    mWorkThread.wait(3000);
}

int AsyncControl::connect(const char * ip, int rock, int slot)
{
    if(plcClient == nullptr)
       return -1;
    int ret = plcClient->ConnectTo(ip, rock, slot);
    if(ret != 0)
    {
        return ret;
    }
    return 0;
}

int AsyncControl::start()
{
    mDropped = 0;
    mCancel = false;
    QTimer::singleShot(0, this, [this](){ if(!this->mControlling) this->startControl(); });
    return 0;
}

int AsyncControl::put(WeldResult data, int db_id)
{   
    mDBID = db_id;
    QMutexLocker lock(&mLock);
    if(mQueue.count() < maxQueuSize)
    {
        mQueue.push(data);
    }
    else
    {
        mQueue.pop();
        mQueue.push(data);
    }     
    wake();
    return 0;
}

bool AsyncControl::judagePower(std::vector<WeldResult> &result, float &power_ratio)
{
    bool send = false;
    int all = result.size();
    // 遍历最后的10个数据
    int count = 10;
    float avg = 0.0;
    if(all > count)
    {   
        float sum = 0.0;
        int valid = 0;
        for(int i = all - count; i < all; i++)
        {
            if(result[i].weld_status < 4)
            {
                int val = result[i].weld_status - 2;
                sum += val;
                valid++;
            }
        }
        avg = sum / valid;
    }

    if(fabs(avg) > 0.5)
    {
        send = true;
    }

    power_ratio = -0.1 * avg;

    return send;
}


void AsyncControl::startControl()
{
    mControlling = true;
    QMutex mut;
    
    mDropped = 0;
    int cmd_id = 0;
    int alarm_frame = 0;
    while(!mCancel)
    {
        mut.lock();
        mStart.wait(&mut);
        mut.unlock();
        LaserControlData task = {0};
        if(mQueue.count() > maxQueuSize - 10)
        {
            std::vector<WeldResult> result;
            {
                QMutexLocker lock(&mLock);
                mQueue.copy(result);
            }
            float power_ratio = 0.0;

            WeldResult last = result.back();
            
            if (judagePower(result, power_ratio) && (last.frame_id - alarm_frame) > 20)
            {
                task.command_id = cmd_id;
                task.power = power_ratio;                
                int result = plcClient->WriteArea(S7AreaDB, mDBID, 0, sizeof(LaserControlData), S7WLByte, &task);                
                if(result != 0)
                {
                    mDropped++;
                }
                alarm_frame = last.frame_id;
                cmd_id++;

                // 获取当前时间戳
                QString timestamp = QTime::currentTime().toString("HH:mm:ss");
                mLog << timestamp.toStdString() << " WriteArea: frame: " << alarm_frame << ", " << task.command_id << ", " << task.power << std::endl;
                // qDebug("WriteArea: frame:%d , %d, %f\n", alarm_frame, task.command_id, task.power);
            }
        }
    }

    if(mDropped > 0)
    {
        qDebug("Dropped %d tasks\n", mDropped);
        mLog << "Dropped: " << mDropped << std::endl;
    }
    mControlling = false;
}

int AsyncControl::disconnect()
{
    stop();
    plcClient->Disconnect();
    return 0;
}

int AsyncControl::stop()
{
    mCancel = true;
    {
        QMutexLocker lock(&mLock);
        mQueue.clear();
    }
    wake();
    return 0;
}
