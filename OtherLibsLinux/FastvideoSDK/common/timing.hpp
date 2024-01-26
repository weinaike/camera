#ifndef __TIMING_HPP__
#define __TIMING_HPP__

///////////////////////////////////////////////////////////////////////////////
// Timing
///////////////////////////////////////////////////////////////////////////////
typedef struct hostTimerStruct *hostTimer_t;
extern hostTimer_t hostTimerCreate();
extern void hostTimerStart(hostTimer_t timer);
extern double hostTimerEnd(hostTimer_t timer);
extern void hostTimerDestroy(hostTimer_t timer);

extern void hostTimerStart(hostTimer_t timer, bool predicate);
extern double hostTimerEnd(hostTimer_t timer, bool predicate);

#endif
