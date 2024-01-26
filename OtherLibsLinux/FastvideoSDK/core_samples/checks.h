#ifndef __FV_CHECKS_H__
#define __FV_CHECKS_H__

#define CHECK_CUDA( statemant ) {	\
	{								\
	  cudaError_t ret_cuda = ( statemant );		\
	  if (ret_cuda != cudaSuccess) {			\
		  fprintf(stderr, "%s\n", cudaGetErrorString(ret_cuda));	\
		  return FAST_EXECUTION_FAILURE;							\
	  }								\
	}								\
}

#define CHECK_FAST( statemant ) {			\
	{										\
		fastStatus_t ret = ( statemant );	\
		if (ret != FAST_OK)					\
			return ret;						\
	}										\
}

#define CHECK_FAST_WITH_MESSAGE( statemant, msg ) {	\
	{										\
		fastStatus_t ret = ( statemant );	\
		if (ret != FAST_OK) { 				\
			fprintf(stderr, msg);			\
			return ret;						\
		}									\
	}										\
}

#define CHECK_CUDA_TO_FAST( statemant ) {	\
	ret_cuda = (statemant);		\
	if (ret_cuda != cudaSuccess) {	\
		return FAST_INTERNAL_ERROR; \
	}								\
}

#define CHECK_FAST_ALLOCATION( statemant ) {	\
	try {										\
		(statemant);							\
	} catch(std::bad_alloc& ba) {				\
		fprintf(stderr, "Memory allocation failed: %s\n", ba.what());	\
		return FAST_INSUFFICIENT_HOST_MEMORY;	\
	}											\
}

#define CHECK_FAST_DEALLOCATION( statemant ) {	\
	CHECK_FAST_WITH_MESSAGE( statemant, "Memory free failed\n" ) \
}

#define CHECK_FAST_OPEN_FILE( statemant ) {	\
	CHECK_FAST_WITH_MESSAGE( statemant, "Input file has unappropriate format\n" ) \
}

#define CHECK_FAST_SAVE_FILE( statemant ) {	\
	CHECK_FAST_WITH_MESSAGE( statemant, "Could not save to file\n" ) \
}


#endif
