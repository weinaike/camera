#include "timing.hpp"

#ifndef __GNUC__
#include <windows.h>

struct hostTimerStruct{
private:
    LARGE_INTEGER _freq, _start, _stop;

public:
    hostTimerStruct(){
        _freq.QuadPart = 0;
        _start.QuadPart = 0;
        _stop.QuadPart = 0;
        QueryPerformanceFrequency(&_freq);
    }

    void hostTimerStart(){
        QueryPerformanceCounter(&_start);
    }

    double hostTimerEnd(){
        QueryPerformanceCounter(&_stop);
        return (double(_stop.QuadPart) - double(_start.QuadPart)) / double(_freq.QuadPart);
    }

    ~hostTimerStruct(){}
};

#else
#include <sys/time.h>
#include <cstdio>

struct hostTimerStruct
{
  double start;
public:
  hostTimerStruct() : start(0.) {};
  void hostTimerStart()
  {
    struct timeval _start;
    gettimeofday(&_start,NULL);
    start = _start.tv_sec*1. +_start.tv_usec*1.e-6;
  }
  double hostTimerEnd()
  {
    struct timeval _stop;
    gettimeofday(&_stop,NULL);
    double stop = _stop.tv_sec*1. +_stop.tv_usec*1.e-6;
    return stop-start;
  }
};
#endif


hostTimer_t hostTimerCreate(){
    return new hostTimerStruct;
}

void hostTimerStart(hostTimer_t timer){
    timer->hostTimerStart();
}

double hostTimerEnd(hostTimer_t timer){
    return timer->hostTimerEnd();
}

void hostTimerDestroy(hostTimer_t timer){
    delete timer;
}


void hostTimerStart(hostTimer_t timer, bool predicate)
{
    if (predicate)
        hostTimerStart(timer);
}

double hostTimerEnd(hostTimer_t timer, bool predicate)
{
    if (predicate)
        return hostTimerEnd(timer);

    return 0.0;
}
