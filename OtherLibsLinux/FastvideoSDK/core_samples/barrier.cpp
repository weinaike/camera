#include "barrier.hpp"

void barrier::wait(void) {
	//fence mechanism
	std::unique_lock<std::mutex> lk(m);
	++counter;
	///while (counter < thread_count);
	++waiting;

	if (counter >= thread_count)
		cv.notify_all();
	else
		cv.wait(lk);
	
	--waiting;
	if (waiting == 0) {
		//reset barrier
		counter = 0;
	}
	lk.unlock();
}
