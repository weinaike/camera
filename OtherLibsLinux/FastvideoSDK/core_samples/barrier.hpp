

#ifndef __BARRIER__
#define __BARRIER__

#include <thread>
#include <atomic>
#include <condition_variable>
#include <mutex>

class barrier {
private:
	std::mutex m;
	std::condition_variable cv;
	volatile int counter;
	int waiting;
	int thread_count;

public:
	barrier(int count) : thread_count(count), counter(0), waiting(0) { }
	void wait(void);
};

#endif // __BARRIER__
